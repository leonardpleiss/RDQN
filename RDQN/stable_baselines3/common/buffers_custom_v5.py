import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import warnings

class PositionalReplayBuffer(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

        if n_envs != 1:
            warnings.warn("The use of this buffer on multiple environments was not tested and may not work properly!")

        assert optimize_memory_usage == False, "Memory optimization is not supported."
        assert optimize_memory_usage == False, "Timeout termination is not supported."

        # Setup timestep storage
        self._current_timestep = np.ones(self.n_envs,)
        self.timesteps = np.zeros((self.buffer_size, self.n_envs))
        self.episode_length = np.zeros((self.buffer_size, self.n_envs))
        self.ep_start_idx = np.ones(self.n_envs,)

        self._max_episode_length = 1

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        self.timesteps[self.pos] = self._current_timestep

        # self.episode_length[self.pos] = self._current_timestep * 2 # Balanced target while episode not done
        # self.episode_length[self.pos] = self._current_timestep # Pure DDQN target while episode not done

        if done.any():

            env_ids = np.where(done)[0]
            ep_lens = self._current_timestep[env_ids].astype(int)
            starts = self.ep_start_idx[env_ids].astype(int)
            ends = np.full_like(starts, self.pos, dtype=int)

            # Compute lengths accounting for wrap
            lengths = (ends - starts + 1) % self.buffer_size
            lengths[lengths <= 0] += self.buffer_size

            # Build all (row, col) index pairs at once
            idx_list = [
                (np.arange(length) + start) % self.buffer_size
                for start, length in zip(starts, lengths)
            ]
            row_idx = np.concatenate(idx_list)
            col_idx = np.repeat(env_ids, lengths)

            self.episode_length[row_idx, col_idx] = np.repeat(ep_lens, lengths)

        self._max_episode_length = max(self._max_episode_length, np.max(self._current_timestep))
        self._current_timestep = np.where(done, 1, self._current_timestep + 1)

        super().add(obs, next_obs, action, reward, done, infos)

        self.ep_start_idx[done] = self.pos

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        upper_bound = self.buffer_size if self.full else self.pos
        
        row_idxes = np.random.randint(0, upper_bound, size=batch_size)
        col_idxes = np.random.randint(0, self.n_envs, size=batch_size)
        batch_idxes = (row_idxes, col_idxes)

        relative_episodic_position = self.timesteps[batch_idxes] / self.episode_length[batch_idxes]

        relative_episodic_position[np.isinf(relative_episodic_position)] = 0. # Pure online target while episode not done

        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        return encoded_sample, batch_idxes, relative_episodic_position
    
    def sample_alternative(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        upper_bound = self.buffer_size if self.full else self.pos
        
        row_idxes = np.random.randint(0, upper_bound, size=batch_size)
        col_idxes = np.random.randint(0, self.n_envs, size=batch_size)
        batch_idxes = (row_idxes, col_idxes)

        subsequent_steps = self.episode_length[batch_idxes] - self.timesteps[batch_idxes]
        subsequent_steps[subsequent_steps<0] = 0. # Pure online target while episode not done

        max_subsequent_steps = self._max_episode_length - 1
        relative_episodic_position = 1 - (subsequent_steps / max_subsequent_steps)

        # print(f"{self.timesteps[batch_idxes]=}")
        # print(f"{self.episode_length[batch_idxes]=}")
        # print(f"{self._max_episode_length=}")
        # print(f"{relative_episodic_position=}")
        # print("---")

        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        return encoded_sample, batch_idxes, relative_episodic_position