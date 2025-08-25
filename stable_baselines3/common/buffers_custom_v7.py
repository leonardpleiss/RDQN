import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import warnings

class SelectiveReplayBuffer(ReplayBuffer):

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

        # Setup selection mask
        self.done_child = np.zeros((self.buffer_size, self.n_envs)).astype(bool)
        self.reward_child = np.zeros((self.buffer_size, self.n_envs)).astype(bool)

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

        self.done_child[self.pos] = done
        self.reward_child[self.pos] = (reward != 0)

        # Update
        self._current_timestep = np.where(done, 1, self._current_timestep + 1)

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample a batch of transitions only from positions where self.sampling_mask is True.
        """

        av_samples = (self.pos * self.n_envs)  if not self.full else (self.buffer_size * self.n_envs)
        reward_samples = (self.rewards != 0).sum()

        reward_slots = int(batch_size * (reward_samples / av_samples))
        done_slots = batch_size - reward_slots

        valid_indices_done = np.argwhere(self.done_child)
        chosen_done = valid_indices_done[np.random.choice(len(valid_indices_done), size=done_slots)]
        row_idxes_done, col_idxes_done = chosen_done[:, 0], chosen_done[:, 1]

        # Select reward samples
        valid_indices_reward = np.argwhere(self.reward_child)
        chosen_reward = valid_indices_reward[np.random.choice(len(valid_indices_reward), size=reward_slots)]
        row_idxes_reward, col_idxes_reward = chosen_reward[:, 0], chosen_reward[:, 1]

        # Select done samples
        row_idxes = np.hstack([row_idxes_reward, row_idxes_done])
        col_idxes = np.hstack([col_idxes_reward, col_idxes_done])

        # Fetch actual samples
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Enable sampling of predecessors
        pre_row_idxes_done = row_idxes_done-1
        pre_row_idxes_done = np.where(row_idxes_done == 0, self.buffer_size-1, pre_row_idxes_done)
        not_first_timestep = self.timesteps[row_idxes_done, col_idxes_done] != 1
        pre_row_idxes_done = pre_row_idxes_done[not_first_timestep]
        pre_col_idxces_done = col_idxes_done[not_first_timestep]
        self.done_child[pre_row_idxes_done, pre_col_idxces_done] = True

        pre_row_idxes_reward = row_idxes_reward-1
        pre_row_idxes_reward = np.where(row_idxes_reward == 0, self.buffer_size-1, pre_row_idxes_reward)
        not_first_timestep = self.timesteps[row_idxes_reward, col_idxes_reward] != 1
        pre_row_idxes_reward = pre_row_idxes_reward[not_first_timestep]
        pre_col_idxces_reward = col_idxes_reward[not_first_timestep]
        self.reward_child[pre_row_idxes_reward, pre_col_idxces_reward] = True

        return encoded_sample
