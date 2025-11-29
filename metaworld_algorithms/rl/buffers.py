import abc, pdb
from collections import deque
from typing import override

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Float

from metaworld_algorithms.types import (
    Action,
    Observation,
    RNNState,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
)


class AbstractReplayBuffer(abc.ABC):
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    @abc.abstractmethod
    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def checkpoint(self) -> ReplayBufferCheckpoint: ...

    @abc.abstractmethod
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None: ...

    @abc.abstractmethod
    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSamples: ...


class ReplayBuffer(AbstractReplayBuffer):
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        self.reset()  # Init buffer

    @override
    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.bit_generator.state,
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.bit_generator.state = ckpt["rng_state"]

    @override
    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        if obs.ndim >= 2:
            assert (
                obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0]
            ), "Batch size must be the same for all transition data."

            # Flatten any batch dims
            flat_obs = obs.reshape(-1, obs.shape[-1])
            flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            flat_action = action.reshape(-1, action.shape[-1])
            flat_reward = reward.reshape(
                -1, 1
            )  # Keep the last dim as 1 for consistency
            flat_done = done.reshape(-1, 1)  # Keep the last dim as 1 for consistency

            # Calculate number of new transitions
            n_transitions = len(flat_obs)

            # Handle buffer wraparound
            indices = np.arange(self.pos, self.pos + n_transitions) % self.capacity

            # Store the transitions
            self.obs[indices] = flat_obs
            self.next_obs[indices] = flat_next_obs
            self.actions[indices] = flat_action
            self.rewards[indices] = flat_reward
            self.dones[indices] = flat_done

            self.pos = (self.pos + n_transitions) % self.capacity
            if self.pos > self.capacity and not self.full:
                self.full = True
        else:
            self.obs[self.pos] = obs.copy()
            self.actions[self.pos] = action.copy()
            self.next_obs[self.pos] = next_obs.copy()
            self.dones[self.pos] = done.copy().reshape(-1, 1)
            self.rewards[self.pos] = reward.copy().reshape(-1, 1)

            self.pos += 1

        if self.pos > self.capacity and not self.full:
            self.full = True
        self.pos %= self.capacity

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        return ReplayBufferSamples(*batch)


class MultiTaskReplayBuffer(AbstractReplayBuffer):
    """Replay buffer for the multi-task benchmarks.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size, num_tasks,).
    When pushing samples to the buffer, the buffer only accepts inputs with batch shape (num_tasks,).
    """

    obs: Float[Observation, "buffer_size task"]
    actions: Float[Action, "buffer_size task"]
    rewards: Float[npt.NDArray, "buffer_size task 1"]
    next_obs: Float[Observation, "buffer_size task"]
    dones: Float[npt.NDArray, "buffer_size task 1"]
    pos: int

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
    ) -> None:
        assert total_capacity % num_tasks == 0, (
            "Total capacity must be divisible by the number of tasks."
        )
        self.capacity = total_capacity // num_tasks
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        # all needed for reward smoothing --> Reggie's original idea about scale and smoothness mattering
        self.max_steps = max_steps
        self.current_trajectory_start = 0

        self.reset(save_rewards=False)  # Init buffer

    @override
    def reset(self, save_rewards=False):
        """Reinitialize the buffer."""
        self.obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.num_tasks, self._action_shape), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0

        if save_rewards:
            self.org_rewards = np.zeros(
                (self.capacity, self.num_tasks, 1), dtype=np.float32
            )
            self.traj_start = 0

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.bit_generator.state,
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.bit_generator.state = ckpt["rng_state"]

    @override
    def add(
        self,
        obs: Float[Observation, " task"],
        next_obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.obs[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.dones[self.pos] = done.copy().reshape(-1, 1)
        self.rewards[self.pos] = reward.reshape(-1, 1).copy()

        self.pos = self.pos + 1
        if self.pos == self.capacity:
            self.full = True

        self.pos = self.pos % self.capacity

    def single_task_sample(self, task_idx: int, batch_size: int) -> ReplayBufferSamples:
        assert task_idx < self.num_tasks, "Task index out of bounds."

        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx][task_idx],
            self.actions[sample_idx][task_idx],
            self.next_obs[sample_idx][task_idx],
            self.dones[sample_idx][task_idx],
            self.rewards[sample_idx][task_idx],
        )

        return ReplayBufferSamples(*batch)

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of size `single_task_batch_size` for each task.

        Args:
            batch_size (int): The total batch size. Must be divisible by number of tasks

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """
        assert batch_size % self.num_tasks == 0, (
            "Batch size must be divisible by the number of tasks."
        )
        single_task_batch_size = batch_size // self.num_tasks

        sample_idx = self._rng.integers(
            low=0,
            high=max(
                self.pos if not self.full else self.capacity, single_task_batch_size
            ),
            size=(single_task_batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        return ReplayBufferSamples(*batch)


class MultiTaskRolloutCollectionBuffer(AbstractReplayBuffer):
    """Collection buffer storing sequences in fixed-size arrays for off-policy learning.

    Unlike the old deque-based approach, this stores sequences in a fixed-size ndarray
    with dimensions [seq_len, buffer_capacity, feature_dim]. When rollouts are longer
    than seq_len, they are stored as sliding windows (stride=1).

    DESIGN:
    - Fixed-size ndarray: stores seq_len+1 observations to get proper next_obs
    - Actions/rewards/dones: seq_len timesteps (transitions)
    - Observations: seq_len+1 timesteps (states) - obs[0:seq_len] and next_obs = obs[1:seq_len+1]
    - Circular buffer: overwrites oldest sequences when full
    - Sliding windows: rollout[0:seq_len+1], rollout[1:seq_len+2], etc.
    - Total storage respects buffer_size limit from OffPolicyTrainingConfig
    """

    # Fixed-size storage arrays
    sequences_obs: np.ndarray  # [buffer_capacity, seq_len, num_tasks, obs_dim]
    sequences_next_obs: np.ndarray  # [buffer_capacity, seq_len, num_tasks, obs_dim]
    sequences_actions: np.ndarray  # [buffer_capacity, seq_len, num_tasks, action_dim]
    sequences_rewards: np.ndarray  # [buffer_capacity, seq_len, num_tasks, 1]
    sequences_dones: np.ndarray  # [buffer_capacity, seq_len, num_tasks, 1]

    buffer_capacity: int  # Max number of sequences to store
    seq_len: int  # Fixed sequence length
    num_tasks: int
    pos: int  # Current write position (circular)
    full: bool  # Whether buffer has wrapped around

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        max_rollout_steps: int = 500,
        seq_len: int = 100,
        seed: int | None = None,
    ) -> None:
        """Initialize the rollout collection buffer.

        Args:
            total_capacity: Maximum total transitions to store (buffer_size from config, e.g., 5e6)
            num_tasks: Number of tasks in the multi-task environment
            env_obs_space: Observation space from the environment
            env_action_space: Action space from the environment
            max_rollout_steps: Maximum timesteps per rollout (default: 500)
            seq_len: Fixed sequence length for stored sequences (from OffPolicyTrainingConfig)
            seed: Random seed for sampling
        """
        self.seq_len = seq_len
        self.num_tasks = num_tasks
        self.max_rollout_steps = max_rollout_steps
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()

        # Calculate buffer capacity: total_capacity / (seq_len * num_tasks)
        # This ensures total storage <= buffer_size
        self.buffer_capacity = total_capacity // (seq_len * num_tasks)
        self.full = False

        # Internal buffer for building current rollout
        self._current_rollout_buffer = MultiTaskRolloutBuffer(
            num_rollout_steps=max_rollout_steps,
            num_tasks=num_tasks,
            env_obs_space=env_obs_space,
            env_action_space=env_action_space,
            seed=seed,
        )

        self.reset()

    @override
    def reset(self) -> None:
        """Reinitialize the buffer."""
        # Initialize fixed-size arrays - separate obs and next_obs for fast sampling
        self.sequences_obs = np.zeros(
            (self.buffer_capacity, self.seq_len, self.num_tasks, self._obs_shape),
            dtype=np.float32
        )
        self.sequences_next_obs = np.zeros(
            (self.buffer_capacity, self.seq_len, self.num_tasks, self._obs_shape),
            dtype=np.float32
        )
        self.sequences_actions = np.zeros(
            (self.buffer_capacity, self.seq_len, self.num_tasks, self._action_shape),
            dtype=np.float32
        )
        self.sequences_rewards = np.zeros(
            (self.buffer_capacity, self.seq_len, self.num_tasks, 1),
            dtype=np.float32
        )
        self.sequences_dones = np.zeros(
            (self.buffer_capacity, self.seq_len, self.num_tasks, 1),
            dtype=np.float32
        )

        self.pos = 0
        self.full = False
        self._current_rollout_buffer.reset()

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        """Save buffer state for checkpointing."""
        # DESIGN: All data now stored in fixed-size ndarrays (no NamedTuples to decompose)
        return {
            "data": {
                # Fixed-size sequence arrays
                "sequences_obs": self.sequences_obs,
                "sequences_next_obs": self.sequences_next_obs,
                "sequences_actions": self.sequences_actions,
                "sequences_rewards": self.sequences_rewards,
                "sequences_dones": self.sequences_dones,
                "pos": int(self.pos),
                "full": bool(self.full),
                # Current rollout buffer arrays and position
                "current_rollout_observations": self._current_rollout_buffer.observations,
                "current_rollout_actions": self._current_rollout_buffer.actions,
                "current_rollout_rewards": self._current_rollout_buffer.rewards,
                "current_rollout_dones": self._current_rollout_buffer.dones,
                "current_rollout_pos": int(self._current_rollout_buffer.pos),
            },
            # RNG states go at top level (saved with JsonSave, not StandardSave)
            "rng_state": {
                "main_rng": self._rng.bit_generator.state,
                "current_rollout_buffer_rng": self._current_rollout_buffer._rng.bit_generator.state,
            },
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        """Load buffer state from checkpoint."""
        for key in ["data", "rng_state"]:
            assert key in ckpt

        # Restore fixed-size sequence arrays
        self.sequences_obs = ckpt["data"]["sequences_obs"]
        self.sequences_next_obs = ckpt["data"]["sequences_next_obs"]
        self.sequences_actions = ckpt["data"]["sequences_actions"]
        self.sequences_rewards = ckpt["data"]["sequences_rewards"]
        self.sequences_dones = ckpt["data"]["sequences_dones"]
        self.pos = ckpt["data"]["pos"]
        self.full = ckpt["data"]["full"]

        # Restore current rollout buffer arrays and position
        self._current_rollout_buffer.observations = ckpt["data"]["current_rollout_observations"]
        self._current_rollout_buffer.actions = ckpt["data"]["current_rollout_actions"]
        self._current_rollout_buffer.rewards = ckpt["data"]["current_rollout_rewards"]
        self._current_rollout_buffer.dones = ckpt["data"]["current_rollout_dones"]
        self._current_rollout_buffer.pos = ckpt["data"]["current_rollout_pos"]

        # Restore RNG states from top-level dict
        self._rng.bit_generator.state = ckpt["rng_state"]["main_rng"]
        self._current_rollout_buffer._rng.bit_generator.state = ckpt["rng_state"]["current_rollout_buffer_rng"]

    @override
    def add(
        self,
        obs: Float[Observation, " task"],
        next_obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
    ) -> None:
        """Add a batch of transitions (one per task) to the current rollout being built.

        Args:
            obs: Current observations for all tasks [num_tasks, obs_dim]
            next_obs: Next observations for all tasks [num_tasks, obs_dim]
            action: Actions for all tasks [num_tasks, action_dim]
            reward: Rewards for all tasks [num_tasks] or [num_tasks, 1]
            done: Done flags for all tasks [num_tasks] or [num_tasks, 1]
        """
        # Add to current rollout buffer
        self._current_rollout_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
        )

        # Check if any episode ended or buffer is full
        if self._current_rollout_buffer.ready or done.any():
            self._store_current_rollout()

    def _store_current_rollout(self) -> None:
        """Store the current rollout as sliding window sequences into fixed arrays.

        SLIDING WINDOW APPROACH:
        - Need seq_len+1 observations for seq_len transitions (to get next_obs)
        - If rollout length <= seq_len: skip (not enough data for proper next_obs)
        - If rollout length > seq_len: store all possible windows with stride=1
          Example: rollout_len=102, seq_len=100 -> store windows at positions 0, 1
                   window[0]: obs[0:100], next_obs[1:101], actions[0:100], rewards[0:100], dones[0:100]
                   window[1]: obs[1:101], next_obs[2:102], actions[1:101], rewards[1:101], dones[1:101]
        """
        if self._current_rollout_buffer.pos == 0:
            return

        rollout_len = self._current_rollout_buffer.pos

        # Skip if rollout is too short: need seq_len+1 observations for seq_len transitions
        if rollout_len <= self.seq_len:
            self._current_rollout_buffer.reset()
            return

        # Calculate number of sliding windows we can extract
        # Each window needs seq_len+1 observations
        num_windows = rollout_len - self.seq_len

        # Extract and store each sliding window
        for window_idx in range(num_windows):
            # Extract observations: [window_idx:window_idx+seq_len+1] for seq_len+1 timesteps
            # This gives us obs[t] for t=0..seq_len, so next_obs[t] = obs[t+1]
            seq_obs = self._current_rollout_buffer.observations[
                window_idx:window_idx + self.seq_len
            ]
            seq_next_obs = self._current_rollout_buffer.observations[
                window_idx + 1:window_idx + self.seq_len + 1
            ]

            # Extract transitions: [window_idx:window_idx+seq_len] for seq_len transitions
            seq_actions = self._current_rollout_buffer.actions[
                window_idx:window_idx + self.seq_len
            ]
            seq_rewards = self._current_rollout_buffer.rewards[
                window_idx:window_idx + self.seq_len
            ]
            seq_dones = self._current_rollout_buffer.dones[
                window_idx:window_idx + self.seq_len
            ]

            # Store in circular buffer at current position
            self.sequences_obs[self.pos] = seq_obs  # [seq_len, num_tasks, obs_dim]
            self.sequences_next_obs[self.pos] = seq_next_obs  # [seq_len, num_tasks, obs_dim]
            self.sequences_actions[self.pos] = seq_actions  # [seq_len, num_tasks, action_dim]
            self.sequences_rewards[self.pos] = seq_rewards  # [seq_len, num_tasks, 1]
            self.sequences_dones[self.pos] = seq_dones  # [seq_len, num_tasks, 1]

            # Advance circular buffer position
            self.pos += 1
            if self.pos >= self.buffer_capacity:
                self.full = True
                self.pos = 0  # Wrap around (circular buffer)

        # Reset for next rollout
        self._current_rollout_buffer.reset()

    def sample(
        self, batch_size: int, seq_len: int | None = None
    ) -> ReplayBufferSamples:
        """Sample sequences from the fixed-size buffer.

        DESIGN: Samples from pre-stored sequences in the fixed-size buffer.
        Each sequence has length self.seq_len with proper next_obs from seq_len+1 stored observations.
        Task information is embedded in the observations (task IDs in last dimensions).

        Args:
            batch_size: Number of sequences to sample
            seq_len: Ignored (for backward compatibility). Uses self.seq_len instead.

        Returns:
            Single ReplayBufferSamples with time dimension:
            - observations: (seq_len, batch_size, obs_dim)
            - actions: (seq_len, batch_size, action_dim)
            - etc.
        """
        # Determine valid range for sampling
        num_available = self.pos if not self.full else self.buffer_capacity

        if num_available == 0:
            raise ValueError("Cannot sample from empty buffer")

        single_task_batch_size = batch_size // self.num_tasks 

        # Randomly sample sequence indices - simple indexing like ReplayBuffer.sample()
        # Shape: (batch_size,)
        seq_indices = self._rng.integers(0, num_available, size=single_task_batch_size)

        mt_batch_size = single_task_batch_size * self.num_tasks
        #batch = map(lambda x: x, batch)
        #batch = map(lambda x: x.reshape(self.seq_len, mt_batch_size, *x.shape[3:]), batch)
        # Do expensive operations ONCE
        obs_batch = self.sequences_obs[seq_indices].transpose(1, 0, 2, 3).reshape(self.seq_len, mt_batch_size, -1)
        actions_batch = self.sequences_actions[seq_indices].transpose(1, 0, 2, 3).reshape(self.seq_len, mt_batch_size, -1)
        next_obs_batch = self.sequences_next_obs[seq_indices].transpose(1, 0, 2, 3).reshape(self.seq_len, mt_batch_size, -1)
        dones_batch = self.sequences_dones[seq_indices].transpose(1, 0, 2, 3).reshape(self.seq_len, mt_batch_size, -1)
        rewards_batch = self.sequences_rewards[seq_indices].transpose(1, 0, 2, 3).reshape(self.seq_len, mt_batch_size, -1)

        # Return single ReplayBufferSamples with time dimension (seq_len, batch, dim)
        # This avoids creating 100 separate Python objects
        return ReplayBufferSamples(
            observations=obs_batch,  # (seq_len, mt_batch_size, obs_dim)
            actions=actions_batch,
            next_observations=next_obs_batch,
            dones=dones_batch,
            rewards=rewards_batch,
        )

        # OLD CODE (backup): Returns list of 100 objects - slower
        # return [
        #     ReplayBufferSamples(
        #         observations=obs_batch[i].astype(np.float32),
        #         actions=actions_batch[i].astype(np.float32),
        #         next_observations=next_obs_batch[i].astype(np.float32),
        #         dones=dones_batch[i].astype(np.float32),
        #         rewards=rewards_batch[i].astype(np.float32),
        #     ) for i in range(self.seq_len)
        # ]

        """
        # Extract sequences using simple indexing
        # sequences_obs shape: [buffer_capacity, seq_len+1, num_tasks, obs_dim]
        # After indexing: [batch_size, seq_len+1, num_tasks, obs_dim]
        sampled_obs_all = self.sequences_obs[seq_indices]

        # Split into obs[t] and next_obs[t] = obs[t+1]
        sampled_obs = sampled_obs_all[:, :-1, :, :]  # [batch_size, seq_len, num_tasks, obs_dim]
        sampled_next_obs = sampled_obs_all[:, 1:, :, :]  # [batch_size, seq_len, num_tasks, obs_dim]

        # Extract actions/rewards/dones with simple indexing
        sampled_actions = self.sequences_actions[seq_indices]
        sampled_rewards = self.sequences_rewards[seq_indices]
        sampled_dones = self.sequences_dones[seq_indices]

        # Reshape to flatten batch and task dimensions: [batch_size, seq_len, num_tasks, dim] -> [batch_size*num_tasks, seq_len, dim]
        # Then transpose to [seq_len, batch_size*num_tasks, dim]
        # Reshape: [batch, seq, tasks, dim] -> [seq, batch*tasks, dim]
        obs_batch = sampled_obs.reshape(batch_size, self.seq_len, -1).transpose(1, 0, 2)
        actions_batch = sampled_actions.reshape(batch_size, self.seq_len, -1).transpose(1, 0, 2)
        next_obs_batch = sampled_next_obs.reshape(batch_size, self.seq_len, -1).transpose(1, 0, 2)
        dones_batch = sampled_dones.reshape(batch_size, self.seq_len, -1).transpose(1, 0, 2)
        rewards_batch = sampled_rewards.reshape(batch_size, self.seq_len, -1).transpose(1, 0, 2)

        # Convert to list of ReplayBufferSamples (one per timestep)
        res = [
            ReplayBufferSamples(
                observations=obs_batch[i].astype(np.float32),
                actions=actions_batch[i].astype(np.float32),
                next_observations=next_obs_batch[i].astype(np.float32),
                dones=dones_batch[i].astype(np.float32),
                rewards=rewards_batch[i].astype(np.float32),
            ) for i in range(self.seq_len)
        ]

        return res 
        """
    
    def get_statistics(self) -> dict[str, float]:
        """Get statistics about stored sequences for monitoring.

        Returns:
            Dictionary with num_sequences, sequence_length, total_transitions
        """
        num_sequences = self.pos if not self.full else self.buffer_capacity

        if num_sequences == 0:
            return {
                "num_sequences": 0,
                "sequence_length": self.seq_len,
                "total_transitions": 0,
            }

        return {
            "num_sequences": num_sequences,
            "sequence_length": self.seq_len,
            "total_transitions": num_sequences * self.seq_len * self.num_tasks,
        }

    def sample_no_seq(self, batch_size: int) -> ReplayBufferSamples:
        """Sample random transitions from stored rollouts.

        Maintains backward compatibility by returning individual transitions.

        Args:
            batch_size: Total number of transitions to sample

        Returns:
            ReplayBufferSamples containing sampled transitions
        """
        if len(self.rollouts) == 0:
            raise ValueError("Cannot sample from empty buffer")

        num_rollouts = len(self.rollouts)
        rollout_indices = self._rng.integers(0, num_rollouts, size=batch_size)

        sampled_obs = []
        sampled_actions = []
        sampled_next_obs = []
        sampled_dones = []
        sampled_rewards = []

        for rollout_idx in rollout_indices:
            rollout = self.rollouts[rollout_idx]
            rollout_len = rollout.observations.shape[0]

            # Sample random timestep and task
            timestep_idx = self._rng.integers(0, rollout_len)
            task_idx = self._rng.integers(0, self.num_tasks)

            # Extract transition
            obs = rollout.observations[timestep_idx, task_idx]
            action = rollout.actions[timestep_idx, task_idx]
            reward = rollout.rewards[timestep_idx, task_idx]
            done = rollout.dones[timestep_idx, task_idx]

            # For next_obs, use next timestep if available
            if timestep_idx + 1 < rollout_len:
                next_obs = rollout.observations[timestep_idx + 1, task_idx]
            else:
                next_obs = obs

            sampled_obs.append(obs)
            sampled_actions.append(action)
            sampled_next_obs.append(next_obs)
            sampled_rewards.append(reward)
            sampled_dones.append(done)

        batch = (
            np.array(sampled_obs, dtype=np.float32),
            np.array(sampled_actions, dtype=np.float32),
            np.array(sampled_next_obs, dtype=np.float32),
            np.array(sampled_dones, dtype=np.float32).reshape(-1, 1),
            np.array(sampled_rewards, dtype=np.float32).reshape(-1, 1),
        )

        return ReplayBufferSamples(*batch)

    def sample_rollouts(
        self, num_rollouts: int, max_length: int | None = None
    ) -> list[Rollout]:
        """Sample complete rollouts from the buffer.

        New capability for algorithms that need sequential data.

        Args:
            num_rollouts: Number of rollouts to sample
            max_length: If specified, truncate rollouts to this length

        Returns:
            List of Rollout objects with sequential data
        """
        if len(self.rollouts) == 0:
            raise ValueError("Cannot sample from empty buffer")

        num_available = len(self.rollouts)
        rollout_indices = self._rng.integers(0, num_available, size=num_rollouts)

        sampled_rollouts = []
        for rollout_idx in rollout_indices:
            rollout = self.rollouts[rollout_idx]

            if max_length is not None:
                current_len = rollout.observations.shape[0]
                if current_len > max_length:
                    rollout = Rollout(
                        observations=rollout.observations[:max_length],
                        actions=rollout.actions[:max_length],
                        rewards=rollout.rewards[:max_length],
                        dones=rollout.dones[:max_length],
                        log_probs=None,
                        means=None,
                        stds=None,
                        values=None,
                        rnn_states=None,
                    )

            sampled_rollouts.append(rollout)

        return sampled_rollouts

class MultiTaskRolloutBuffer:
    num_rollout_steps: int
    num_tasks: int
    pos: int

    observations: Float[Observation, "timestep task"]
    actions: Float[Action, "timestep task"]
    rewards: Float[npt.NDArray, "timestep task 1"]
    dones: Float[npt.NDArray, "timestep task 1"]

    values: Float[npt.NDArray, "timestep task 1"]
    log_probs: Float[npt.NDArray, "timestep task 1"]
    means: Float[Action, "timestep task"]
    stds: Float[Action, "timestep task"]
    rnn_states: Float[RNNState, "timestep task"] | None = None

    def __init__(
        self,
        num_rollout_steps: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        rnn_state_dim: int | None = None,
        dtype: npt.DTypeLike = np.float32,
        seed: int | None = None,
    ) -> None:
        self.num_rollout_steps = num_rollout_steps
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self._rnn_state_dim = rnn_state_dim
        self.dtype = dtype
        self.reset()  # Init buffer

    def reset(self) -> None:
        """Reinitialize the buffer."""
        self.observations = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._obs_shape), dtype=self.dtype
        )
        self.actions = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._action_shape),
            dtype=self.dtype,
        )
        self.rewards = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype
        )
        self.dones = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype
        )

        self.log_probs = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype
        )
        self.values = np.zeros_like(self.rewards)
        self.means = np.zeros_like(self.actions)
        self.stds = np.zeros_like(self.actions)

        if self._rnn_state_dim is not None:
            self.rnn_states = np.zeros(
                (self.num_rollout_steps, self.num_tasks, self._rnn_state_dim),
                dtype=self.dtype,
            )

        self.pos = 0

    @property
    def ready(self) -> bool:
        return self.pos == self.num_rollout_steps

    def add(
        self,
        obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
        value: Float[npt.NDArray, " task"] | None = None,
        log_prob: Float[npt.NDArray, " task"] | None = None,
        mean: Float[Action, " task"] | None = None,
        std: Float[Action, " task"] | None = None,
        rnn_state: Float[RNNState, " task"] | None = None,
    ):
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.dones[self.pos] = done.copy().reshape(-1, 1)

        if value is not None:
            self.values[self.pos] = value.copy()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            self.means[self.pos] = mean.copy()
        if std is not None:
            self.stds[self.pos] = std.copy()
        if rnn_state is not None:
            assert self.rnn_states is not None
            self.rnn_states[self.pos] = rnn_state.copy()

        self.pos += 1

    def get(
        self,
    ) -> Rollout:
        return Rollout(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.means,
            self.stds,
            self.values,
            self.rnn_states,
        )
