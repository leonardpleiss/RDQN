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
        self.max_episode_length = 1.

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
        self.max_timesteps = np.zeros((self.buffer_size, self.n_envs))

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
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + new_td_errors

        self.max_timesteps[self.pos] = self._current_timestep * 2
        self.max_episode_length = max(self.max_episode_length, np.max(self._current_timestep))

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            self._max_sum_td = max(self._max_sum_td, np.max(self.sum_tds[ep_done_row_idxes, ep_done_col_idxes]))

            self.max_timesteps[ep_done_row_idxes, ep_done_col_idxes] = self.timesteps[self.pos][ep_done_col_idxes]

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
        sum_tds = self.lagging_sum_tds[batch_idxes].copy()
        is_curr_episode = self.episodes[batch_idxes] == self._current_episode
        sum_tds[is_curr_episode] = self._max_sum_td

        cum_tds = self.lagging_cum_tds[batch_idxes]

        subsequent_tds = sum_tds - cum_tds

        # reliability = (1 - (subsequent_tds / max_sum_tds)) ** self._alpha2
        reliability = (1 - (subsequent_tds / sum_tds)) ** self._alpha2

        # relative_episodic_position = self.timesteps[batch_idxes] / self.max_episode_length
        relative_episodic_position = self.timesteps[batch_idxes] / self.max_timesteps[batch_idxes]

        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        assert (subsequent_tds >= 0).all(), f"{subsequent_tds[subsequent_tds<0], cum_tds[subsequent_tds<0], sum_tds[subsequent_tds<0]}"
        assert (relative_episodic_position <= 1).all()
        assert (relative_episodic_position >= 0).all()

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), reliability, (row_idxes, col_idxes), self._max_td, subsequent_tds, relative_episodic_position # self.max_ep_return, self.rmean_ep_return
    
    
    def update_priorities(self, idxes, new_td_errors):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        new_td_errors = new_td_errors[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]

        # Mask relevant variables
        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - new_td_errors

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update, kind="table")

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        sum_deltas = deltas * (episodes_to_update != self._current_episode)
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, sum_deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])

        self._max_td = max(self._max_td, np.max(new_td_errors))
        self._max_sum_td = np.max(self.cum_tds)

        self.counter += 1

    def update_reliabilities(self):

        self.lagging_sum_tds = self.sum_tds
        self.lagging_cum_tds = self.cum_tds
