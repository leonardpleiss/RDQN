import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import logging
import numpy as np

class R_UNI(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            log_path: str = "",
            alpha2: float = .6,
            check_frequency: int = 100_000,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

        assert n_envs == 1, "Multiple environments currently not supported."

        self.counter = 0

        self._alpha2 = np.float64(alpha2)

        self._max_td = 1e-6 # 1.
        self._max_sum_td = 1.
        self.max_ep_return = 1.
        self.rmean_ep_return = 1.

        # Setup sampling array
        self.sample_arange = np.arange(self.buffer_size * self.n_envs)

        # Setup episode storage
        self._current_episode = np.arange(1, self.n_envs+1, dtype=int)
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=int)
        self.episodes_played = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # Setup td error, td sum and td cumsum storage
        self.last_done = np.zeros(self.n_envs, dtype=np.bool_)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.max_sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.td_errors = np.zeros((self.buffer_size, self.n_envs))
        self.reliabilities = np.zeros((self.buffer_size, self.n_envs))

        # Setup timestep storage
        self._current_timestep = np.ones(self.n_envs,)
        self.timesteps = np.zeros((self.buffer_size, self.n_envs))

        # Setup logger
        self.log_path = log_path
        self.logger = logging.getLogger(__name__)

        # logging.basicConfig(filename=self.log_path + 'replay_value_prediction_model.log', encoding='utf-8', level=logging.DEBUG)
        self.logger.info(f"{self.buffer_size=}")
        self.logger.info(f"{self.n_envs=}")
        self.logger.info(f"{self.device=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        new_td_errors: List[Dict[str, Any]],
    ) -> None:
        
        # Add transition information
        self.td_errors[self.pos] = new_td_errors
        self.reliabilities[self.pos] = 1.
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done
        self._max_td = max(self._max_td, np.max(new_td_errors))
        self._max_sum_td = max(self._max_sum_td, self.cum_tds[self.pos])

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        upper_bound = self.buffer_size if self.full else self.pos
        
        row_idxes = np.random.randint(0, upper_bound, size=batch_size)
        col_idxes = np.random.randint(0, self.n_envs, size=batch_size)
        batch_idxes = row_idxes, col_idxes

        # Mask relevant variables
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        reliabilities = self.reliabilities[batch_idxes]

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), reliabilities, (row_idxes, col_idxes), self._max_td # self.max_ep_return, self.rmean_ep_return
    
    
    def update_priorities(self, idxes, new_td_errors):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        new_td_errors = new_td_errors[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]

        deltas = self.td_errors[row_idxes, col_idxes] - new_td_errors

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas
        self._max_td = max(self._max_td, np.max(new_td_errors))
        self._max_sum_td = np.max(self.cum_tds)

        self.counter += 1

    def update_reliabilities(self):

        filled_elements = self.pos if not self.full else self.buffer_size
        self.td_errors = self.td_errors.ravel()
        self.episodes = self.episodes.ravel()

        first_ep = self.episodes[0]
        last_ep = self.episodes[-1]
        overlap = first_ep == last_ep
        
        if overlap:
            overlap_mask = self.episodes[::-1] == last_ep
            num_overlapping_transitions = np.argmax(~overlap_mask)

            self.td_errors = np.concatenate([self.td_errors[-num_overlapping_transitions:], self.td_errors[:-num_overlapping_transitions]])
            self.episodes = np.concatenate([self.episodes[-num_overlapping_transitions:], self.episodes[:-num_overlapping_transitions]])

        # Episode sums
        _, group_idx = np.unique(self.episodes, return_inverse=True)
        episode_sums = np.zeros(len(group_idx), dtype=self.td_errors.dtype)
        
        np.add.at(episode_sums, group_idx, self.td_errors)
        self.sum_tds = episode_sums[group_idx]
       
        # Episode cumsums
        self.cum_tds = np.zeros_like(self.td_errors)
        for eid in group_idx:
            mask = self.episodes == eid
            self.cum_tds[mask] = np.cumsum(self.td_errors[mask])

        # MAYBE SET SUM FOR CURR EPISODE TO MAXSUM

        if overlap:
            self.td_errors = np.concatenate([self.td_errors[num_overlapping_transitions:], self.td_errors[:num_overlapping_transitions]])
            self.episodes = np.concatenate([self.td_errors[num_overlapping_transitions:], self.td_errors[:num_overlapping_transitions]])

        self.td_errors = self.td_errors.reshape(-1, 1)
        self.episodes = self.episodes.reshape(-1, 1)
        self.sum_tds = self.sum_tds.reshape(-1, 1)
        self.cum_tds = self.cum_tds.reshape(-1, 1)
        self.reliabilities[:filled_elements] = (self.sum_tds[:filled_elements] - self.cum_tds[:filled_elements]) / self.sum_tds[:filled_elements]

        # print(f"{self.td_errors=}")
        # print(f"{self.cum_tds=}")
        # print(f"{self.sum_tds=}")
        # print(f"{self.episodes=}")
        # print(f"{self._current_episode=}")
        # print(f"{overlap=}")

        # mask = self.cum_tds > self.sum_tds
        # print(self.sum_tds[mask])
        # print(self.cum_tds[mask])
        # print(self.episodes[mask])
        # print(self.timesteps[mask])

        assert (self.sum_tds[:filled_elements] >= self.cum_tds[:filled_elements]).all()
        assert (self.reliabilities[:filled_elements] >= 0).all()