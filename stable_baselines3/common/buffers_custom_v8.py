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
        self.signal_mask = np.zeros((self.buffer_size, self.n_envs)).astype(bool)

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

        self.signal_mask[self.pos] = done or (reward != 0)

        # Update
        self._current_timestep = np.where(done, 1, self._current_timestep + 1)

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample a batch of transitions only from positions where self.sampling_mask is True.
        """

        av_samples = (self.pos * self.n_envs)  if not self.full else (self.buffer_size * self.n_envs)
        signal_set_size = self.signal_mask.sum()

        signal_slots = int(batch_size * (signal_set_size / av_samples))
        noise_slots = batch_size - signal_slots

        valid_indices_signal = np.argwhere(self.signal_mask)
        chosen_signal = valid_indices_signal[np.random.choice(len(valid_indices_signal), size=signal_slots)]
        row_idxes_signal, col_idxes_signal = chosen_signal[:, 0], chosen_signal[:, 1]

        row_idxes_noise = np.random.randint(av_samples, size=noise_slots)
        col_idxes_noise = np.random.randint(self.n_envs, size=noise_slots)

        # Select done samples
        row_idxes = np.hstack([row_idxes_noise, row_idxes_signal])
        col_idxes = np.hstack([col_idxes_noise, col_idxes_signal])

        # Fetch actual samples
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Enable sampling of predecessors
        pre_row_idxes_signal = row_idxes_signal-1
        pre_row_idxes_signal = np.where(row_idxes_signal == 0, self.buffer_size-1, pre_row_idxes_signal)
        not_first_timestep = self.timesteps[row_idxes_signal, col_idxes_signal] != 1
        pre_row_idxes_signal = pre_row_idxes_signal[not_first_timestep]
        pre_col_idxces_signal = col_idxes_signal[not_first_timestep]
        self.signal_mask[pre_row_idxes_signal, pre_col_idxces_signal] = True

        return encoded_sample
