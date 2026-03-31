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

    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
        previous_features: Float[Array, "batch feature_dim"] | None = None,
    ) -> tuple[Self, LogDict, Float[Array, "batch feature_dim"] | None]:
        key, critic_loss_key = jax.random.split(self.key)

        # Sample a'
        if self.split_critic_losses:
            next_actions, next_action_log_probs = jax.vmap(
                lambda x: self.actor.apply_fn(self.actor.params, x).sample_and_log_prob(
                    seed=critic_loss_key
                )
            )(data.observations)
            q_values = jax.vmap(self.critic.apply_fn, in_axes=(None, 0, 0))(
                self.critic.target_params, data.next_observations, next_actions
            )
        else:
            next_actions, next_action_log_probs = self.actor.apply_fn(
                self.actor.params, data.next_observations
            ).sample_and_log_prob(seed=critic_loss_key)
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

        def critic_loss(
            combined_params: dict | FrozenDict,
            _data: ReplayBufferSamples,
            _q_values: Float[Array, "#batch 1"],
            _alpha_val: Float[Array, "#batch 1"],
            _next_action_log_probs: Float[Array, " #batch"],
            _previous_features: Float[Array, "batch feature_dim"] | None,  # NEW: Pass previous features explicitly
            _task_weights: Float[Array, "#batch 1"] | None = None,
        ) -> tuple[Float[Array, ""], tuple]:
            """Compute critic loss with optional LSTM masking.

            DESIGN: Accepts combined parameters dict for joint critic+LSTM optimization
            RETURN: (loss, (q_pred_mean, h_new, c_new, intermediate_features)) - includes LSTM states in auxiliary

            Args:
                combined_params: Either {'critic': params, 'lstm': params} or just critic params
                _data: Batch of transitions
                _q_values: Target Q-values from target network
                _alpha_val: Temperature parameter values
                _next_action_log_probs: Log probabilities of next actions
                _previous_features: Previous timestep's features for LSTM input (None on first step)
                _task_weights: Optional task-specific weights

            Returns:
                (loss, auxiliary) where auxiliary = (q_pred_mean, h_new, c_new, intermediate_features)
            """
            # Handle both combined and single params (backward compatibility)
            # DESIGN: If params is dict with 'critic' key, extract critic and LSTM params
            # Otherwise, assume it's just critic params (no LSTM)
            if isinstance(combined_params, dict) and 'critic' in combined_params:
                critic_params = combined_params['critic']
                lstm_params = combined_params.get('lstm', None)
            else:
                critic_params = combined_params
                lstm_params = None

            # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
            min_qf_next_target = jnp.min(
                _q_values, axis=0
            ) - _alpha_val * _next_action_log_probs.reshape(-1, 1)

            next_q_value = jax.lax.stop_gradient(
                _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
            )

            # Forward pass with or without LSTM
            if lstm_params is not None and hasattr(self, 'lstm'):
                # ===== FIRST STEP HANDLING =====
                # DESIGN: On first step, _previous_features is None - use all-ones mask
                # This allows the network to perform a normal forward pass and populate features
                # Subsequent steps will use LSTM-generated masks based on previous features
                if _previous_features is None:
                    # First step: use all-ones mask (no modulation)
                    # Infer batch size from observations
                    batch_size = _data.observations.shape[0]
                    # Get MOORE width from LSTM output_size
                    moore_width = self.lstm.params['params']['W_out'].shape[1]
                    mask = jnp.ones((batch_size, moore_width))
                    h_new, c_new = None, None
                else:
                    # ===== LSTM TEMPORAL PROCESSING =====
                    # DESIGN: LSTM receives previous network features and produces mask
                    # INPUT: _previous_features = previous timestep's masked features
                    # STATE: self.lstm_h, self.lstm_c = current LSTM states (updated each step)
                    # OUTPUT: mask for current timestep's features
                    mask, h_new, c_new = self.lstm.apply_fn(
                        lstm_params,
                        _previous_features,  # Previous network intermediate state (passed as arg)
                        self.lstm_h,         # Current hidden state from class attribute
                        self.lstm_c          # Current cell state from class attribute
                    )

                # ===== CRITIC FORWARD PASS WITH MASKING =====
                # CRITICAL: mutable=['intermediates'] required to return updated_vars
                # CRITICAL: store=True tells MOORENetwork to save features via self.variable()
                # NOTE: critic_params already has {'params': ...} structure from TrainState
                # NOTE: Pass mask and store as kwargs to avoid vmap issues
                q_pred, updated_vars = self.critic.apply_fn(
                    critic_params,               # Already has {'params': ...} structure
                    _data.observations,
                    _data.actions,
                    mask,                   # LSTM-generated mask (or all-ones on first step) - pass as kwarg
                    True,                   # Tell MOORENetwork to store intermediates
                    mutable=['intermediates'],   # Capture modified collections
                )

                # ===== EXTRACT MASKED FEATURES FOR NEXT TIMESTEP =====
                # CRITICAL: Cannot append to self.grin_state_vars here because we're inside jax.grad()
                # JAX doesn't allow side effects (mutating Python state) inside traced functions
                # Instead, return the features as auxiliary output and append outside
                # STRUCTURE: updated_vars['intermediates']['VmapQValueFunction_0']['MOORENetwork_0']['or']
                # Shape: (num_critics, batch, width) - we'll use first critic's features
                intermediate_features = updated_vars['intermediates']['VmapQValueFunction_0']['MOORENetwork_0']['or'][0]
            else:
                # ===== NO LSTM PATH (backward compatibility) =====
                # Standard critic forward pass without masking or temporal processing
                q_pred = self.critic.apply_fn(
                    critic_params,
                    _data.observations,
                    _data.actions
                )
                h_new, c_new = None, None
                intermediate_features = None

            if self.max_q_value is not None:
                # HACK: Clipping Q values to approximate theoretical maximum for Metaworld
                next_q_value = jnp.clip(
                    next_q_value, -self.max_q_value, self.max_q_value
                )
                q_pred = jnp.clip(q_pred, -self.max_q_value, self.max_q_value)

            if _task_weights is not None:
                loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
            else:
                loss = ((q_pred - next_q_value) ** 2).mean()

            # Return loss and auxiliary (includes new LSTM states and intermediate features)
            # auxiliary = (q_pred_mean, h_new, c_new, intermediate_features)
            return loss, (q_pred.mean(), h_new, c_new, intermediate_features)

        # ===== COMBINE PARAMETERS FOR JOINT OPTIMIZATION =====
        # DESIGN: Create a single pytree containing both critic and LSTM parameters
        # JAX's value_and_grad will compute gradients for all params in this structure
        # BENEFIT: Single backward pass computes gradients for both networks
        if hasattr(self, 'lstm'):
            combined_params = {
                'critic': self.critic.params,  # Critic network parameters
                'lstm': self.lstm.params       # LSTM network parameters
            }
        else:
            # Backward compatibility: if no LSTM, just use critic params
            combined_params = self.critic.params

        if self.split_critic_losses:
            # ===== JOINT GRADIENT COMPUTATION (per-task split) =====
            # GRADIENT COMPUTATION: jax.value_and_grad differentiates w.r.t. first arg
            # has_aux=True: loss function returns (loss, auxiliary_data)
            # vmap: vectorize over tasks for per-task loss computation
            (critic_loss_value, aux), combined_grads = jax.vmap(
                jax.value_and_grad(critic_loss, has_aux=True),
                in_axes=(None, 0, 0, 0, 0, None, 0),  # previous_features not vmapped (broadcast to all tasks)
                out_axes=0,
            )(
                combined_params,  # Gradients computed for this pytree
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                previous_features,  # NEW: Pass previous features
                task_weights,
            )

            # ===== EXTRACT AUXILIARY OUTPUTS =====
            # aux is tuple: (q_pred.mean(), h_new, c_new)
            qf_values = aux[0]  # Q-value predictions (for logging)
            h_new = aux[1]      # New LSTM hidden state
            c_new = aux[2]      # New LSTM cell state

            # ===== UPDATE LSTM STATES FOR NEXT TIMESTEP =====
            # CRITICAL: Store updated states as class attributes
            # These will be used in the next timestep's critic_loss call
            # TEMPORAL CONTINUITY: h_new/c_new from timestep t become inputs for t+1
            if h_new is not None:
                self = self.replace(lstm_h=h_new)
            if c_new is not None:
                self = self.replace(lstm_c=c_new)

            # ===== EXTRACT GRADIENTS FROM COMBINED STRUCTURE =====
            # combined_grads has same structure as combined_params:
            # If LSTM: {'critic': critic_grads, 'lstm': lstm_grads}
            # If no LSTM: just critic_grads
            if isinstance(combined_grads, dict):
                critic_grads = combined_grads['critic']
            else:
                critic_grads = combined_grads
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
            )
        else:
            (critic_loss_value, aux), combined_grads = jax.value_and_grad(
                critic_loss, has_aux=True
            )(
                combined_params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                previous_features,  # NEW: Pass previous features
                task_weights,
            )
            # Extract auxiliary outputs
            qf_values = aux[0]
            h_new = aux[1]
            c_new = aux[2]
            intermediate_features = aux[3]

            # Update LSTM states for next timestep
            if h_new is not None:
                self = self.replace(lstm_h=h_new)
            if c_new is not None:
                self = self.replace(lstm_c=c_new)

            # NOTE: Don't append here - still inside JIT boundary
            # Will return intermediate_features and append outside JIT

            # Handle gradients
            if isinstance(combined_grads, dict):
                critic_grads = combined_grads['critic']
            else:
                critic_grads = combined_grads
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

        key, optimizer_key = jax.random.split(key)

        # Apply critic gradients
        critic = self.critic.apply_gradients(
            grads=critic_grads,
            optimizer_extra_args={
                "task_losses": critic_loss_value,
                "key": optimizer_key,
            },
        )
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )

        # Apply LSTM gradients (if LSTM exists and gradients were computed)
        if hasattr(self, 'lstm') and isinstance(combined_grads, dict) and 'lstm' in combined_grads:
            lstm_grads = combined_grads['lstm']
            lstm = self.lstm.apply_gradients(
                grads=lstm_grads,
                optimizer_extra_args={
                    "task_losses": critic_loss_value,
                    "key": optimizer_key,
                }
            )
            self=self.replace(lstm=lstm)
        flat_params_crit, _ = flatten_util.ravel_pytree(critic.params)

        return self.replace(critic=critic, key=key), {
            "losses/qf_values": qf_values.mean(),
            "losses/qf_loss": critic_loss_value.mean(),
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }, intermediate_features

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

    grin_state_vars: list = struct.field(default_factory=list)  # Stores network intermediate states (masked features)
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
            total_capacity=self.rollout_capacity,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            max_rollout_steps=self.max_rollout_steps,
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
            previous_features = None if len(self.grin_state_vars) == 0 else self.grin_state_vars[-1]

            # Call JIT-compiled function
            self, logs, intermediate_features = self._update_inner(d, previous_features)

            # ===== APPEND INTERMEDIATE FEATURES OUTSIDE JIT BOUNDARY =====
            # CRITICAL: Python list mutation happens here, outside @jax.jit
            # This allows proper JIT compilation of _update_inner
            if intermediate_features is not None:
                self.grin_state_vars.append(intermediate_features)

        return self, logs
    
    @override
    def update(self, data: list[ReplayBufferSamples]) -> tuple[Self, LogDict]:
        # 1. PREPARE DATA: Stack the list of samples into a single PyTree with time dimension
        # Input: list of T objects, each with shape (B, D)
        # Output: One object with shape (T, B, D)
        stacked_data = jax.tree.map(lambda *xs: jnp.stack(xs), *data)
        
        # Determine dimensions
        total_steps = stacked_data.observations.shape[0]
        batch_size = stacked_data.observations.shape[1]
        
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
            # Handle grin_state_vars for consistency
            if features_0 is not None:
                # We convert to a list here to match your original API expectation
                # though keeping it as a JAX array is usually better for performance.
                current_self.grin_state_vars.append(features_0)
            return current_self, logs_0

        # 4. STEP 1 to T: Use jax.lax.scan for the rest
        # We slice the data to get steps 1 through T
        scan_data = jax.tree.map(lambda x: x[1:], stacked_data)
        
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

        # 5. AGGREGATE RESULTS
        # Combine Step 0 features with Scan features (Steps 1..T)
        # features_0 shape: (Batch, Dim) -> expand to (1, Batch, Dim)
        # scan_features_history shape: (T-1, Batch, Dim)
        if features_0 is not None:
            features_0_expanded = jnp.expand_dims(features_0, 0)
            all_features = jnp.concatenate([features_0_expanded, scan_features_history], axis=0)
            
            # Update the grin_state_vars with the full history
            # NOTE: Ideally store this as an array, but if you need a list:
            # final_self.grin_state_vars.extend([all_features[i] for i in range(total_steps)])
            # For pure JAX performance, prefer storing the array:
            final_self.grin_state_vars.append(all_features)
        
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