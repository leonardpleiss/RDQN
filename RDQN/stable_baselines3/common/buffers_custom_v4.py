import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import logging
import numpy as np

class DR_UNI(ReplayBuffer):

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
        self.max_subsequent = 0.
        self.max_discounted_return = 1.

        # Setup sampling array
        self.sample_arange = np.arange(self.buffer_size * self.n_envs)

        # Setup episode storage
        self._current_episode = np.arange(1, self.n_envs+1, dtype=int)
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=int)
        self.episodes_played = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # Setup td error, td sum and td cumsum storage
        self.last_done = np.zeros(self.n_envs,dtype=np.bool_)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.max_sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.td_errors = np.zeros((self.buffer_size, self.n_envs))

        # Get lagging structures
        self.lagging_sum_tds = np.ones((self.buffer_size, self.n_envs))
        self.lagging_cum_tds = np.ones((self.buffer_size, self.n_envs))

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
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep

        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)

            discounted_return = np.sum(
                self.rewards[ep_done_row_idxes, ep_done_col_idxes] * 
                .99 ** self.timesteps[ep_done_row_idxes, ep_done_col_idxes])
            
            self.max_discounted_return = max(self.max_discounted_return, discounted_return)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, discount_factor = float) -> ReplayBufferSamples:

        upper_bound = self.buffer_size if self.full else self.pos
        
        # 1. Sample a batch of (episode, timestep) indices
        row_idxes = np.random.randint(0, upper_bound, size=batch_size)
        col_idxes = np.random.randint(0, self.n_envs, size=batch_size)
        batch_idxes = (row_idxes, col_idxes)

        episodes = self.episodes[batch_idxes]
        timesteps = self.timesteps[batch_idxes]

        played_and_change_mask = np.isin(self.episodes, episodes)

        filtered_episodes = self.episodes[played_and_change_mask]
        filtered_timesteps = self.timesteps[played_and_change_mask]
        filtered_tds = self.td_errors[played_and_change_mask]#  ** 2
        
        episode_mask = (filtered_episodes[:, None] == episodes[None, :])
        timestep_mask = (filtered_timesteps[:, None] > timesteps[None, :])
        
        cum_mask = episode_mask & timestep_mask
        
        timesteps_diff = filtered_timesteps[:, None] - timesteps[None, :]

        steps_until_end = np.where(cum_mask, timesteps_diff, 0)
               
        discounts = discount_factor ** steps_until_end
        
        td_values_to_sum = np.where(cum_mask, filtered_tds[:, None] * discounts, 0)
    
        discounted_subsequent_errors = np.sum(td_values_to_sum, axis=0)

        self.max_subsequent = max(self.max_subsequent, np.max(discounted_subsequent_errors))

        reliability = 1 - (discounted_subsequent_errors / self.max_subsequent)

        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), reliability, (row_idxes, col_idxes), self.max_discounted_return, discounted_subsequent_errors

    def update_priorities(self, idxes, new_td_errors):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        new_td_errors = new_td_errors[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]

        # Mask relevant variables
        deltas = self.td_errors[row_idxes, col_idxes] - new_td_errors

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

    def update_reliabilities(self):
        pass
