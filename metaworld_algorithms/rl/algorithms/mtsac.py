"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import pdb
import dataclasses
from functools import partial
from typing import Self, override

import flax.linen as nn
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import LSTMConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from metaworld_algorithms.nn.moore import GatedRecurrentMask
from metaworld_algorithms.optim.pcgrad import PCGradState
from metaworld_algorithms.rl.buffers import (
    MultiTaskReplayBuffer,
    MultiTaskRolloutCollectionBuffer,
)
from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    Ensemble,
    QValueFunction,
)
from metaworld_algorithms.types import (
    Action,
    LogDict,
    Observation,
    ReplayBufferSamples,
)

from .base import OffPolicyAlgorithm
from .utils import TrainState


class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


@jax.jit
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False
    max_q_value: float | None = 5000


class MTSAC(OffPolicyAlgorithm[MTSACConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)
    split_actor_losses: bool = struct.field(pytree_node=False)
    split_critic_losses: bool = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    max_q_value: float | None = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: MTSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTSAC":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key = (
            jax.random.split(master_key, 4)
        )

        actor_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.actor_config
        )
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_init_key, dummy_obs),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        alpha_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        dummy_task_ids = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_init_key, dummy_task_ids),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()
        
        return MTSAC(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            num_critics=config.num_critics,
            split_actor_losses=config.actor_config.network_config.optimizer.requires_split_task_losses,
            split_critic_losses=config.critic_config.network_config.optimizer.requires_split_task_losses,
            max_q_value=config.max_q_value,
        )

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> MultiTaskReplayBuffer:
        return MultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.actor, observations))

    def split_data_by_tasks(
        self,
        data: PyTree[Float[Array, "batch data_dim"]],
        task_ids: Float[npt.NDArray, "batch num_tasks"],
    ) -> PyTree[Float[Array, "num_tasks per_task_batch data_dim"]]:
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(
            leaf: Float[Array, "batch data_dim"],
        ) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data), sorted_indices

    def unsplit_data_by_tasks(
        self,
        split_data: PyTree[Float[Array, "num_tasks per_task_batch data_dim"]],
        sort_indices: jax.Array,
    ) -> PyTree[Float[Array, "batch data_dim"]]:
        def reconstruct_leaf(
            leaf: Float[Array, "num_tasks per_task_batch data_dim"],
        ) -> Float[Array, "batch data_dim"]:
            batch_size = leaf.shape[0] * leaf.shape[1]
            flat = leaf.reshape(batch_size, leaf.shape[-1])
            # Create inverse permutation
            inverse_indices = jnp.zeros_like(sort_indices)
            inverse_indices = inverse_indices.at[sort_indices].set(
                jnp.arange(batch_size)
            )
            return flat[inverse_indices]

        return jax.tree.map(reconstruct_leaf, split_data)

    def _compute_targets(
        self, 
        data: ReplayBufferSamples, 
        alpha_val: Array, 
        key: PRNGKeyArray
    ) -> Array:
        """Helper to compute target Q-values. Handles the split/non-split logic via vmap."""
        
        # Define single-sample/batch logic
        def get_q_target(p_actor, p_critic, obs, next_obs, r, d):
            # 1. Sample Next Action
            next_dist = self.actor.apply_fn(p_actor, next_obs)
            next_action, next_log_prob = next_dist.sample_and_log_prob(seed=key)
            
            # 2. Get Target Q
            # Note: We assume target_params has same structure as params
            target_q = self.critic.apply_fn(p_critic, next_obs, next_action)
            
            # 3. Soft Q Calc
            min_q = jnp.min(target_q, axis=0) - alpha_val * next_log_prob.reshape(-1, 1)
            return jax.lax.stop_gradient(r + (1 - d) * self.gamma * min_q)

        if self.split_critic_losses:
            # Vectorize over the 'task' dimension implicit in the batch or params
            # Assuming data is split by task if split_losses is True
            return jax.vmap(get_q_target, in_axes=(None, None, 0, 0, 0, 0))(
                self.actor.params, self.critic.target_params, 
                data.observations, data.next_observations, data.rewards, data.dones
            )
        else:
            return get_q_target(
                self.actor.params, self.critic.target_params,
                data.observations, data.next_observations, data.rewards, data.dones
            )

    def _critic_loss_fn(
        self,
        params: dict,  # Expects {'critic': ..., 'lstm': ...}
        obs: Array,
        actions: Array,
        target_q: Array,
        previous_features: Array | None,
        task_weights: Array | None,
        lstm_h: Array,
        lstm_c: Array
    ):
        """Pure loss function. No side effects. Returns loss + aux tuple."""
        critic_params = params['critic']
        lstm_params = params.get('lstm', None)

        # --- 1. LSTM Mask Generation ---
        # We handle the 'Step 0' logic (previous_features=None) cleanly
        has_lstm = lstm_params is not None
        
        def apply_lstm_mask(_prev, _h, _c):
            return self.lstm.apply_fn(lstm_params, _prev, _h, _c)

        def default_mask(_bs: int, _h: Array, _c: Array) -> tuple[Array, Array, Array]:
            """Generates an all-ones mask and returns the input h/c states unchanged."""
            # Get MOORE width from LSTM output_size
            width = self.lstm.params['params']['W_out'].shape[1]
            mask = jnp.ones((_bs, width))
            # FIX: Return the input states (_h, _c) as placeholders for the next step
            return mask, _h, _c
        
        # Use functional conditions if we are inside JIT, 
        # but since 'previous_features is None' is a static trace-time property in the scan loop,
        # Python if is actually correct and faster here.
        if has_lstm and previous_features is not None:
            mask, h_new, c_new = apply_lstm_mask(previous_features, lstm_h, lstm_c)
        elif has_lstm:
             # First step or no features yet
            batch_size = obs.shape[0]
            
            # FIX: Pass lstm_h and lstm_c to default_mask
            mask, h_new, c_new = default_mask(batch_size, lstm_h, lstm_c)
        else:
            # No LSTM at all
            mask, h_new, c_new = None, None, None
        
        # --- 2. Critic Forward ---
        # We always pass mask. If mask is None (No LSTM), the network should handle it 
        # (or we check has_lstm). Assuming MOORENetwork handles optional mask or we conditionally pass it.
        if has_lstm:
            q_pred, updated_vars = self.critic.apply_fn(
                critic_params, obs, actions, 
                mask, True, mutable=['intermediates']
            )
            # Extract features for next step
            # Shape: (num_critics, batch, width) -> Take [0]
            curr_features = jax.lax.stop_gradient(updated_vars['intermediates']['VmapQValueFunction_0']['MOORENetwork_0']['or'][0])
            h_new = jax.lax.stop_gradient(h_new)
            c_new = jax.lax.stop_gradient(c_new)
        else:
            q_pred = self.critic.apply_fn(critic_params, obs, actions)
            curr_features = None
        
        # --- 3. Loss Calculation ---
        # Handle Max Q Clipping
        if self.max_q_value is not None:
            q_pred = jnp.clip(q_pred, -self.max_q_value, self.max_q_value)

        # MSE
        diff = (q_pred - target_q) ** 2
        
        # Apply weights (Use 1.0 default to avoid if/else)
        weights = task_weights if task_weights is not None else 1.0
        loss = (weights * diff).mean()

        return loss, (q_pred.mean(), h_new, c_new, curr_features)

    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
        previous_features: Float[Array, "batch feature_dim"] | None = None,
    ) -> tuple[Self, LogDict, Float[Array, "batch feature_dim"] | None]:
        
        # 1. Setup Keys & Targets
        key, new_key = jax.random.split(self.key)
        target_q_values = self._compute_targets(data, alpha_val, key)

        # 2. Prepare Parameters
        # Always use the combined dictionary structure. 
        # Even if LSTM is missing, 'lstm': None is safer than changing structure.
        combined_params = {'critic': self.critic.params}
        if hasattr(self, 'lstm') and self.lstm is not None:
            combined_params['lstm'] = self.lstm.params

        # 3. Define Gradient Function
        # We bind the 'static' arguments (data, targets) so the grad function only sees params
        def loss_wrapper(p):
            return self._critic_loss_fn(
                p, data.observations, data.actions, target_q_values,
                previous_features, task_weights, self.lstm_h, self.lstm_c
            )

        # 4. Compute Gradients
        if self.split_critic_losses:
            # vmap the gradient function over the task dimension
            # We assume data is already split/shaped correctly for vmap if this flag is set
            grad_fn = jax.vmap(jax.value_and_grad(loss_wrapper, has_aux=True))
            (loss_val, aux), grads = grad_fn(combined_params)
            
            # Flatten grads for metrics (taking mean across tasks)
            # This handles the 'dict' structure of grads automatically
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), grads['critic'])
            )
        else:
            grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
            (loss_val, aux), grads = grad_fn(combined_params)
            flat_grads, _ = flatten_util.ravel_pytree(grads['critic'])

        # 5. Extract Auxiliary Outputs
        qf_mean, h_new, c_new, intermediate_features = aux
        
        # 6. Apply Updates
        # -- Critic --
        # Note: If split_losses, loss_val is a vector.
        critic_grads = grads['critic']
        new_critic = self.critic.apply_gradients(
            grads=critic_grads,
            optimizer_extra_args={"task_losses": loss_val, "key": key}
        )
        # Soft update targets
        new_critic = new_critic.replace(
            target_params=optax.incremental_update(
                new_critic.params, new_critic.target_params, self.tau
            )
        )

        # -- LSTM --
        new_lstm = self.lstm
        if 'lstm' in grads and grads['lstm'] is not None:
            new_lstm = self.lstm.apply_gradients(
                grads=grads['lstm'],
                optimizer_extra_args={"task_losses": loss_val, "key": key}
            )

        # 7. Update Self (Functional State Update)
        # Create new self, updating LSTM states if they exist
        new_self = self.replace(
            critic=new_critic,
            lstm=new_lstm,
            key=new_key,
            # Update h/c if they were returned (meaning LSTM ran)
            lstm_h=h_new if h_new is not None else self.lstm_h,
            lstm_c=c_new if c_new is not None else self.lstm_c
        )

        # 8. Logs
        # Handle mean vs vector loss for logging
        loss_scalar = jnp.mean(loss_val)
        
        logs = {
            "losses/qf_values": jnp.mean(qf_mean),
            "losses/qf_loss": loss_scalar,
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
        }

        return new_self, logs, intermediate_features

    def update_actor(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "batch 1"],
        task_weights: Float[Array, "batch 1"] | None = None,
    ) -> tuple[Self, Float[Array, " batch"], LogDict]:
        key, actor_loss_key = jax.random.split(self.key)

        def actor_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _alpha_val: Float[Array, "batch 1"],
            _task_weights: Float[Array, "batch 1"] | None = None,
        ):
            action_samples, log_probs = self.actor.apply_fn(
                params, _data.observations
            ).sample_and_log_prob(seed=actor_loss_key)
            log_probs = log_probs.reshape(-1, 1)

            q_values = self.critic.apply_fn(
                self.critic.params, _data.observations, action_samples
            )
            min_qf_values = jnp.min(q_values, axis=0)
            if _task_weights is not None:
                loss = (task_weights * (_alpha_val * log_probs - min_qf_values)).mean()
            else:
                loss = (_alpha_val * log_probs - min_qf_values).mean()
            return loss, log_probs

        if self.split_actor_losses:
            (actor_loss_value, log_probs), actor_grads = jax.vmap(
                jax.value_and_grad(actor_loss, has_aux=True),
                in_axes=(None, 0, 0, 0),
                out_axes=0,
            )(self.actor.params, data, alpha_val, task_weights)
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
            )
        else:
            (actor_loss_value, log_probs), actor_grads = jax.value_and_grad(
                actor_loss, has_aux=True
            )(self.actor.params, data, alpha_val, task_weights)
            flat_grads, _ = flatten_util.ravel_pytree(actor_grads)

        key, optimizer_key = jax.random.split(key)
        actor = self.actor.apply_gradients(
            grads=actor_grads,
            optimizer_extra_args={
                "task_losses": actor_loss_value,
                "key": optimizer_key,
            },
        )

        flat_params_act, _ = flatten_util.ravel_pytree(actor.params)
        logs = {
            "losses/actor_loss": actor_loss_value.mean(),
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
        }

        return (self.replace(actor=actor, key=key), log_probs, logs)

    def update_alpha(
        self,
        log_probs: Float[Array, " batch"],
        task_ids: Float[npt.NDArray, " batch num_tasks"],
    ) -> tuple[Self, LogDict]:
        def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
            log_alpha: jax.Array
            log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
            return (-log_alpha * (log_probs + self.target_entropy)).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            self.alpha.params
        )
        alpha = self.alpha.apply_gradients(grads=alpha_grads)

        return self.replace(alpha=alpha), {
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportArgumentType]
        }

    @jax.jit
    def _update_inner(
        self,
        data: ReplayBufferSamples,
        previous_features: Float[Array, "batch feature_dim"] | None = None
    ) -> tuple[Self, LogDict, Float[Array, "batch feature_dim"] | None]:
        task_ids = data.observations[..., -self.num_tasks :]

        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        if self.use_task_weights:
            task_weights = extract_task_weights(self.alpha.params, task_ids)
        else:
            task_weights = None

        actor_data = critic_data = data
        actor_alpha_vals = critic_alpha_vals = alpha_vals
        actor_task_weights = critic_task_weights = task_weights
        alpha_val_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, _ = self.split_data_by_tasks(data, task_ids)
            split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(
                alpha_vals, task_ids
            )
            split_task_weights, _ = (
                self.split_data_by_tasks(task_weights, task_ids)
                if task_weights is not None
                else (None, None)
            )

            if self.split_critic_losses:
                critic_data = split_data
                critic_alpha_vals = split_alpha_vals
                critic_task_weights = split_task_weights

            if self.split_actor_losses:
                actor_data = split_data
                actor_alpha_vals = split_alpha_vals
                actor_task_weights = split_task_weights

        self, critic_logs, intermediate_features = self.update_critic(
            critic_data, critic_alpha_vals, critic_task_weights, previous_features
        )
        self, log_probs, actor_logs = self.update_actor(
            actor_data, actor_alpha_vals, actor_task_weights
        )
        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)
        self, alpha_logs = self.update_alpha(log_probs, task_ids)

        # HACK: PCGrad logs
        assert isinstance(self.critic.opt_state, tuple)
        assert isinstance(self.actor.opt_state, tuple)
        critic_optim_logs = (
            {
                f"metrics/critic_{key}": value
                for key, value in self.critic.opt_state[0]._asdict().items()
            }
            if isinstance(self.critic.opt_state[0], PCGradState)
            else {}
        )
        actor_optim_logs = (
            {
                f"metrics/actor_{key}": value
                for key, value in self.actor.opt_state[0]._asdict().items()
            }
            if isinstance(self.actor.opt_state[0], PCGradState)
            else {}
        )

        return self, {
            **critic_logs,
            **actor_logs,
            **alpha_logs,
            **critic_optim_logs,
            **actor_optim_logs,
        }, intermediate_features

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        # No LSTM for base MTSAC, so previous_features is always None
        self, logs, _ = self._update_inner(data, previous_features=None)
        return self, logs

@dataclasses.dataclass(frozen=True)
class MTSACSequentialConfig(MTSACConfig):
    """Config for MTSAC with sequential rollout collection buffer.

    Inherits all MTSAC config options, only changes the buffer type.
    Uses MultiTaskRolloutCollectionBuffer instead of MultiTaskReplayBuffer.
    """
    rollout_capacity: int = 2000  # Number of rollouts to store
    max_rollout_steps: int = 500  # Max steps per rollout
    lstm_config: LSTMConfig | None = None  # LSTM configuration for temporal learning


class MTSACSequential(MTSAC):
    """MTSAC variant that uses sequential rollout collection buffer with LSTM support.

    This variant stores complete rollouts/trajectories instead of individual transitions,
    preserving temporal structure within episodes. Includes LSTM for learning temporal
    patterns and generating masks for the critic network.
    """

    rollout_capacity: int = struct.field(pytree_node=False)
    max_rollout_steps: int = struct.field(pytree_node=False)
    batch_size: int = 6400

    ## ===== LSTM-RELATED ATTRIBUTES =====
    # DESIGN: Store LSTM state and temporal information as class attributes
    # This enables temporal continuity across timesteps during training

    grin_state_vars: Array | None = None #list = struct.field(default_factory=list)  # Stores network intermediate states (masked features)
                                                                 # USAGE: grin_state_vars[-1] fed to LSTM at each timestep
                                                                 # GROWS: Appends new masked features after each forward pass

    lstm: TrainState | None = None  # LSTM TrainState (params, optimizer state, apply_fn)
                                     # UPDATED: Gradients applied via lstm.apply_gradients()

    lstm_h: Array | None = None  # Current LSTM hidden state (batch, hidden_size)
                                  # UPDATED: Set to h_new after each critic_loss call
                                  # TEMPORAL: Carries information across timesteps

    lstm_c: Array | None = None  # Current LSTM cell state (batch, hidden_size)
                                  # UPDATED: Set to c_new after each critic_loss call
                                  # TEMPORAL: Long-term memory across timesteps

    @override
    @staticmethod
    def initialize(
        config: MTSACSequentialConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTSACSequential":
        """Initialize MTSACSequential with LSTM support.

        DESIGN: MTSACSequential REQUIRES lstm_config - will fail if not provided
        This is intentional to ensure LSTM is always initialized for this variant.
        """
        # Call MTSAC.initialize and add sequential-specific fields
        mtsac = MTSAC.initialize(config, env_config, seed)

        # ===== LSTM CONFIGURATION REQUIREMENTS =====
        # DESIGN CHOICE: Fail fast if lstm_config is missing
        # RATIONALE: MTSACSequential is specifically for LSTM-based temporal learning
        assert hasattr(config, 'lstm_config'), "MTSACSequential requires lstm_config in config"
        assert config.lstm_config is not None, "lstm_config cannot be None for MTSACSequential"

        # ===== CREATE LSTM NETWORK =====
        # PARAMETERS:
        # - hidden_size: LSTM internal state dimension (e.g., 64)
        # - output_size: Mask dimension (MUST match MOORE network width)
        lstm_net = GatedRecurrentMask(
            hidden_size=config.lstm_config.hidden_size,
            output_size=config.lstm_config.output_size
        )

        # ===== INITIALIZE LSTM PARAMETERS =====
        # CAUTION: Use different seed from main network to avoid correlation
        key = jax.random.PRNGKey(seed + 1000)  # Offset seed for LSTM
        dummy_input = jnp.ones((1, config.lstm_config.output_size))  # Dummy for init
        lstm_params = lstm_net.init(key, dummy_input)

        # ===== CREATE LSTM TRAINSTATE =====
        # TrainState bundles: params, optimizer state, apply_fn
        # OPTIMIZER: Specified in lstm_config (can be different from critic optimizer)
        lstm = TrainState.create(
            apply_fn=lstm_net.apply,
            params=lstm_params,
            tx=config.lstm_config.optimizer.spawn()  # Create optimizer instance
        )

        # ===== INITIALIZE LSTM STATES FROM LEARNED h0/c0 =====
        # DESIGN: Start with learned initial states (h0, c0 are trainable parameters)
        # These will be updated to h_new, c_new after first forward pass
        # BROADCAST: Tile to batch size for batch processing
        batch_size = 6400 
        lstm_h = jnp.tile(lstm_params['params']['h0'][None, :], (batch_size, 1))
        lstm_c = jnp.tile(lstm_params['params']['c0'][None, :], (batch_size, 1))

        return MTSACSequential(
            **{k: getattr(mtsac, k) for k in mtsac.__dataclass_fields__.keys()},
            rollout_capacity=config.rollout_capacity,
            max_rollout_steps=config.max_rollout_steps,
            batch_size=batch_size,
            lstm=lstm,
            lstm_h=lstm_h,
            lstm_c=lstm_c,
        )

    def reset_lstm_h_c_states(self, batch_size) -> Self:
        lstm_h = jnp.tile(self.lstm.params['params']['h0'][None, :], (batch_size, 1))
        lstm_c = jnp.tile(self.lstm.params['params']['c0'][None, :], (batch_size, 1))
        self=self.replace(lstm_h=lstm_h)
        self=self.replace(lstm_c=lstm_c)
        return self 
    
    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> MultiTaskRolloutCollectionBuffer:
        """Spawn sequential rollout collection buffer instead of standard replay buffer."""
        return MultiTaskRolloutCollectionBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            max_rollout_steps=self.max_rollout_steps,
            seq_len=config.seq_len,
            seed=seed,
        )
    
    @override
    def update_classic(self, data: list[ReplayBufferSamples]) -> tuple[Self, LogDict]:
        batch_size = data[0].observations.shape[0] if len(data) > 1 else 6400
        self=self.reset_lstm_h_c_states(batch_size)
        for d in data:
            # ===== GET PREVIOUS FEATURES OUTSIDE JIT BOUNDARY =====
            # CRITICAL: Access Python list here, before calling JIT-compiled function
            # On first step: None (use all-ones mask)
            # On subsequent steps: Last stored features from grin_state_vars
            previous_features = None if self.grin_state_vars is None else self.grin_state_vars

            # Call JIT-compiled function
            self, logs, intermediate_features = self._update_inner(d, previous_features)

            # ===== APPEND INTERMEDIATE FEATURES OUTSIDE JIT BOUNDARY =====
            # CRITICAL: Python list mutation happens here, outside @jax.jit
            # This allows proper JIT compilation of _update_inner
            if intermediate_features is not None:
                self = self.replace(grin_state_vars=intermediate_features)

        return self, logs

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        # NEW: data is already a single ReplayBufferSamples with shape (seq_len, batch, dim)
        # No need to stack - buffer.sample() returns the right format
        stacked_data = data

        # Determine dimensions
        total_steps = stacked_data.observations.shape[0]
        batch_size = stacked_data.observations.shape[1]
        print(f"sequence length: {total_steps}")

        # OLD CODE (backup): When buffer.sample() returned list[ReplayBufferSamples]
        # stacked_data = jax.tree.map(lambda *xs: jnp.stack(xs), *data)
        # total_steps = stacked_data.observations.shape[0]
        # batch_size = stacked_data.observations.shape[1]
        
        # 2. INITIALIZE: Reset LSTM states (h0, c0) for the new batch
        current_self = self.reset_lstm_h_c_states(batch_size)
        
        # 3. STEP 0: Run the first step manually
        # We do this because step 0 requires previous_features=None, which triggers
        # specific logic (all-ones mask) in your critic_loss.
        first_step_data = jax.tree.map(lambda x: x[0], stacked_data)
        
        current_self, logs_0, features_0 = current_self._update_inner(
            first_step_data, 
            previous_features=None
        )
        
        # If we only have one step, return early
        if total_steps == 1:
            return current_self, logs_0

        scan_data = jax.tree.map(lambda x: x[1:], stacked_data)
        
        @jax.checkpoint
        def scan_step(carry, x):
            # Unpack carry
            step_self, prev_features = carry
            
            # Execute update
            # Note: step_self contains the updated lstm_h/lstm_c from the previous step
            new_self, step_logs, new_features = step_self._update_inner(
                x, 
                previous_features=prev_features
            )
            
            # Pack carry and output
            # We output (step_logs, new_features) to stack them over time
            return (new_self, new_features), (step_logs, new_features)

        # Run the optimized loop
        # carry_init is the result of Step 0
        carry_init = (current_self, features_0)
        
        (final_self, final_features), (scan_logs, scan_features_history) = jax.lax.scan(
            scan_step, 
            carry_init, 
            scan_data
        )
        
        # Combine logs
        # We need to average the logs from Step 0 and the logs from Scan
        combined_logs = {}
        for k, v0 in logs_0.items():
            # v0 is scalar (mean over batch for step 0)
            # scan_logs[k] is array of shape (T-1,)
            
            # Concatenate [v0] and scan_logs[k]
            v_scan = scan_logs[k]
            # Ensure v0 is 1D array for concatenation if necessary
            all_values = jnp.concatenate([jnp.array([v0]), v_scan])
            combined_logs[k] = jnp.mean(all_values)

        return final_self, combined_logs