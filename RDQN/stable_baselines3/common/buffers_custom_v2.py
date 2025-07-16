import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import logging
import numpy as np

class CustomPrioritizedReplayBufferCumSum(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = 1.,
            check_frequency: int = 100_000,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

        self.debug_mode = debug_mode
        self._alpha = np.float64(alpha)
        self._alpha2 = np.float64(alpha2)

        self.check_counter = 0
        self.check_frequency = check_frequency

        # Setup sampling array
        self.sample_arange = np.arange(self.buffer_size * self.n_envs)

        # Setup episode storage
        self._current_episode = np.arange(1, self.n_envs+1, dtype=int)
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=int)
        self.episodes_played = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # Setup td error, td sum and td cumsum storage
        self._max_td = 1
        self.last_done = np.zeros(self.n_envs,dtype=np.bool_)
        self._cum_td = np.zeros(self.n_envs,)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.td_errors = np.zeros((self.buffer_size, self.n_envs))
        self.sampling_weights = np.zeros((self.buffer_size, self.n_envs))

        # Setup timestep storage
        self._current_timestep = np.ones(self.n_envs,)
        self.timesteps = np.zeros((self.buffer_size, self.n_envs))

        # Setup logger
        self.log_path = log_path
        self.logger = logging.getLogger(__name__)
        # logging.basicConfig(filename=self.log_path + 'replay_value_prediction_model.log', encoding='utf-8', level=logging.DEBUG)

        self.logger.info(f"{self.buffer_size=}")
        self.logger.info(f"{self._alpha=}")
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
    ) -> None:
        
        if self.debug_mode:
            assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

        self.td_errors[self.pos] = self._max_td
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep
        
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + self._max_td
        self.sampling_weights[self.pos] = self.td_errors[self.pos] ** self._alpha # Preliminary sampling weights

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)
    
    def sample(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):

        assert beta > 0
        
        if self.debug_mode:
            assert self.sampling_weights.min() >= 0

        num_transitions_gathered = (self.pos if not self.full else self.buffer_size) * self.n_envs

        # Get rows
        row_weights = np.cumsum(self.sampling_weights[:num_transitions_gathered, :].sum(axis=1))
        sampling_weight_sum = row_weights[-1]
        sample_vals = np.random.rand(batch_size) * sampling_weight_sum
        row_idxes = np.searchsorted(row_weights, sample_vals)

        # Get columns
        if self.n_envs > 1:
            col_vals = row_weights[row_idxes] - sample_vals
            col_weights = np.cumsum(self.sampling_weights[row_idxes, :], axis=1)
            col_idxes = np.argmax(col_weights>col_vals.reshape(-1,1),axis=1)
        else:
            col_idxes = np.zeros(shape=(batch_size,), dtype=np.int64)

        # Encode
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Get importance sampling weights
        sampling_weight_sum = row_weights[-1]
        sampling_probas_of_batch = self.sampling_weights[row_idxes, col_idxes] / sampling_weight_sum
        p_min = self.sampling_weights[:num_transitions_gathered].min() / sampling_weight_sum # sampling_probas_of_batch.min()
        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        IS_weights = (sampling_probas_of_batch * num_transitions_gathered) ** (-beta) / max_weight

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)

    def sample_naive(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):

        assert beta > 0
        assert self.n_envs == 1
        
        if self.debug_mode:
            assert self.sampling_weights.min() >= 0

        sampling_weight_sum = self.sampling_weights.sum()
        num_transitions_gathered = (self.pos if not self.full else self.buffer_size) * self.n_envs
        sampling_probas = self.sampling_weights.flatten() / sampling_weight_sum
        row_idxes = np.random.choice(self.sample_arange, p=sampling_probas, size=batch_size)
        col_idxes = np.zeros(shape=(batch_size,), dtype=np.int64)

        # Encode
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Get importance sampling weights
        IS_weights = []
        p_min = sampling_probas.min()
        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        sampling_probas_of_batch = sampling_probas[row_idxes]
        IS_weights = (sampling_probas_of_batch * num_transitions_gathered) ** (-beta) / max_weight

        # print(
        #     np.hstack([
        #         self.episodes, self.timesteps, self.td_errors, self.cum_tds, self.sum_tds, self.sampling_weights
        #     ])
        # )

        # print(f"{sampling_probas_of_batch=}")
        # print(f"{p_min=}")
        # print(f"{max_weight=}")
        # print(f"{beta=}")
        # print(f"{row_idxes=}")
        # print(f"{IS_weights=}")

        # if self.full & (self.pos == 10):
        #     import sys
        #     sys.exit()
        # print("---")

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)
    
    def update_priorities(self, idxes, priorities):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        priorities = priorities[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - priorities

        # Obtaining change masks
        # played_and_change_mask2 = copy.copy(self.episodes_played)
        # sum_change_mask = np.isin(self.episodes[played_and_change_mask2], episodes_to_update)
        # played_and_change_mask2[played_and_change_mask2] = sum_change_mask

        played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        # self.full_check()
        self.update_sampling_weights(played_and_change_mask)

    def update_sampling_weights(self, played_mask=None):
                
        if self.debug_mode:
            assert self.td_errors.min() >= 0
            assert self.sum_tds.min() >= 0
            assert self.cum_tds.min() >= 0

        is_current_episode = np.isin(self.episodes[played_mask], self._current_episode, kind="table")

        self.sampling_weights[played_mask] = np.where(
            is_current_episode,
            self.td_errors[played_mask] ** self._alpha,
            (((self.cum_tds[played_mask] / self.sum_tds[played_mask]) + .5) ** self._alpha2) * (self.td_errors[played_mask] ** self._alpha)
        )

    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):

        rel_weights = ((self.cum_tds[ep_done_row_idxes, ep_done_col_idxes] / self.sum_tds[ep_done_row_idxes, ep_done_col_idxes]) + .5) ** self._alpha2
        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = rel_weights * (self.td_errors[ep_done_row_idxes, ep_done_col_idxes] ** self._alpha)


    def full_check(self):
        
        if not self.debug_mode:
            return
        
        self.check_counter += 1
        if not self.check_counter%self.check_frequency==0:
            return

        played_and_done = (self.episodes != 0) & (self.episodes != self._current_episode)

        # Sampling weights
        self.logger.info(f"-------------- BUFFER REPORT: TS{self.check_counter} --------------")
        buffer_fill = 1. if self.full else np.round(self.pos/self.buffer_size, 8)
        self.logger.info(f"Buffer fill: {buffer_fill}")
        self.logger.info(f"td_errors: {np.percentile(self.td_errors[played_and_done], [0, 5, 50, 95, 100])}")
        self.logger.info(f"cum_tds: {np.percentile(self.cum_tds[played_and_done], [0, 5, 50, 95, 100])}")
        self.logger.info(f"sum_tds: {np.percentile(self.sum_tds[played_and_done], [0, 5, 50, 95, 100])}")
        self.logger.info(f"sampling_weights: {np.percentile(self.sampling_weights[played_and_done], [0, 5, 50, 95, 100])}")
        self.logger.info("---------------------------------------------------")

        assert np.min(self.td_errors[played_and_done]) >= 0
        assert np.min(self.cum_tds[played_and_done]) >= 0
        assert np.min(self.sum_tds[played_and_done]) > 0

        assert np.isfinite(self.td_errors).all()
        assert np.isfinite(self.cum_tds).all()
        assert np.isfinite(self.sum_tds).all()
        assert np.isfinite(self.sampling_weights).all()

        # Check cumulative sums & sums
        eps_played_and_done = np.unique(self.episodes[played_and_done])
        for ep in eps_played_and_done:
            ep_mask = self.episodes == ep
            if (ep != self.episodes[-1]) & (self.timesteps[ep_mask][0] == 1): 
                assert np.isclose(self.cum_tds[ep_mask], np.cumsum(self.td_errors[ep_mask])).all(), f"{ep}, {self.td_errors[ep_mask]=}, {self.cum_tds[ep_mask]=}, {self.sum_tds[ep_mask]=}"
                assert np.isclose(self.sum_tds[ep_mask], np.sum(self.td_errors[ep_mask])).all(), f"{ep}, {self.td_errors[ep_mask]=}, {self.cum_tds[ep_mask]=}, {self.sum_tds[ep_mask]=}"
            assert np.isclose(self.sum_tds[ep_mask], self.cum_tds[ep_mask][-1]).all(), f"{ep}, {self.td_errors[ep_mask]=}, {self.cum_tds[ep_mask]=}, {self.sum_tds[ep_mask]=}"

        print("All sums and cumulative sums seem to be accurate.")

class CustomPrioritizedReplayBufferCumSum2(CustomPrioritizedReplayBufferCumSum):
    
    def __init__(        
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        alpha: float = .6,
        log_path: str = "",
        debug_mode: bool = True,
        alpha2: float = 1.,
        check_frequency: int = 100_000,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
        )

        self._max_sum_follow_td = 1.

    def update_sampling_weights(self, played_mask=None): # 3.4

                
        if self.debug_mode:
            assert self.td_errors.min() >= 0
            assert self.sum_tds.min() >= 0
            assert self.cum_tds.min() >= 0

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        is_current_episode = np.isin(self.episodes[played_mask], self._current_episode, kind="table")

        td_errors = self.td_errors[played_mask]
        weight_current_episode = td_errors ** self._alpha
        weight_other_episodes = weight_current_episode * (1 - (subsequent_tds[played_mask] / self._max_sum_follow_td)) ** self._alpha2

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = np.where(is_current_episode, weight_current_episode, weight_other_episodes)
        
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability_discount = subsequent_tds / self._max_sum_follow_td 
        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = self.td_errors[ep_done_row_idxes, ep_done_col_idxes] ** self._alpha * (1 - reliability_discount) ** self._alpha2

class CustomPrioritizedReplayBuffer(CustomPrioritizedReplayBufferCumSum):

    def update_sampling_weights(self):
        self.sampling_weights = self.td_errors ** self._alpha

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        
        if self.debug_mode:
            assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

        self.td_errors[self.pos] = self._max_td
        self.sampling_weights[self.pos] = self._max_td ** self._alpha

        super().add(obs, next_obs, action, reward, done, infos)

    def update_priorities(self, idxes, priorities):
        row_idxes, col_idxes = idxes
        deltas = self.td_errors[row_idxes, col_idxes] - priorities
        self.td_errors[row_idxes, col_idxes] -= deltas
        self.update_sampling_weights()


class CustomPrioritizedReplayBufferCumSumProp(CustomPrioritizedReplayBufferCumSum):

    """
    Herausforderung: Propagation resultiert darin, dass sich Deltas verschieben
    Ansatz: Man müsste eine Doppelstruktur schaffen, sodass Deltas nur auf gemessenen (und nicht auf propagierten Errors) basieren
    """

    def update_priorities(self, idxes, priorities):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(row_idxes) >= 0
            assert np.max(row_idxes) < len(self.observations)
            assert np.min(col_idxes) >= 0
            assert np.max(col_idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0), f"{np.where(self.cum_tds[ep_done_mask] < 0)=}"
                assert np.min(self.sum_tds[ep_done_mask] >= 0), f"{np.where(self.sum_tds[ep_done_mask] < 0)=}"
                assert np.min(self.td_errors[ep_done_mask] >= 0), f"{np.where(self.td_errors[ep_done_mask] < 0)=}"

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        if len(unique_idx) != len(priorities):
            priorities = priorities[unique_idx]
            row_idxes = row_idxes[unique_idx]
            col_idxes = col_idxes[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        curr_ts_deltas = self.td_errors[row_idxes, col_idxes] - priorities
        prev_ts_deltas = abs(curr_ts_deltas) * (episodes_to_update == self.episodes[row_idxes-1, col_idxes])
        deltas = curr_ts_deltas - prev_ts_deltas
        
        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= curr_ts_deltas
        self.td_errors[row_idxes-1, col_idxes] += prev_ts_deltas
        self.cum_tds[row_idxes-1, col_idxes] += prev_ts_deltas

        if (deltas!=0).any():

            deltas_larger_zero_mask = deltas!=0
            deltas_larger_zero = deltas[deltas_larger_zero_mask]
            episodes_with_delta = episodes_to_update[deltas_larger_zero_mask]
            timesteps_to_update_with_delta = timesteps_to_update[deltas_larger_zero_mask]

            # Obtaining change masks
            played_and_change_mask = np.isin(self.episodes, episodes_to_update)

            # Updating sums            
            sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_with_delta
            sum_deltas = sum_mask * deltas_larger_zero
            sum_delta = sum_deltas.sum(axis=1)
            new_sum_tds = self.sum_tds[played_and_change_mask] - sum_delta
            self.sum_tds[played_and_change_mask] = new_sum_tds

            # Updating cumulative sums
            cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update_with_delta)
            cum_deltas = cum_mask * deltas_larger_zero
            cum_delta = cum_deltas.sum(axis=1)
            new_cum_tds = self.cum_tds[played_and_change_mask] - cum_delta
            self.cum_tds[played_and_change_mask] = new_cum_tds

        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        #self.full_check()
        self.update_sampling_weights()



class CustomPrioritizedReplayBufferCumSum3(CustomPrioritizedReplayBufferCumSum):

    """
    Uses alpha to regularize the entire priorization, not only the error - does not use alpha2
    """
    
    def __init__(        
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        alpha: float = .6,
        log_path: str = "",
        debug_mode: bool = True,
        alpha2: float = 1.,
        check_frequency: int = 100_000,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        self._max_sum_follow_td = 1.

    def update_sampling_weights(self, played_mask=None):
                
        is_current_episode = np.isin(self.episodes[played_mask], self._current_episode, kind="table")

        if self.debug_mode:

            not_current_episode_mask = ~is_current_episode
            assert self.td_errors[played_mask][not_current_episode_mask].min() >= 0
            assert self.sum_tds[played_mask][not_current_episode_mask].min() >= 0
            assert self.cum_tds[played_mask][not_current_episode_mask].min() >= 0

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        td_errors = self.td_errors[played_mask]
        weight_current_episode = td_errors ** self._alpha
        weight_other_episodes = (td_errors * (1 - (subsequent_tds[played_mask] / self._max_sum_follow_td))) ** self._alpha

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = np.where(is_current_episode, weight_current_episode, weight_other_episodes)
        
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability_discount = subsequent_tds / self._max_sum_follow_td 
        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = (self.td_errors[ep_done_row_idxes, ep_done_col_idxes] * (1 - reliability_discount)) ** self._alpha




class CustomPrioritizedReplayBufferCumSum4(CustomPrioritizedReplayBufferCumSum):

    """
    with reliability estimates for unfinished runs
    """

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = 1.,
            check_frequency: int = 100_000,
    ):
        
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        self._max_sum_follow_td = 1.

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        if self.debug_mode:
            assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

        self.td_errors[self.pos] = self._max_td
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep
        
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + self._max_td
        self.sum_tds[self.pos] = self._max_sum_follow_td

        self.sampling_weights[self.pos] = self._max_td ** self._alpha # Initiate with max priority & reliability of 1

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)

    def update_sampling_weights(self, played_mask=None): # 3.4
                
        if self.debug_mode:
            assert self.td_errors.min() >= 0
            assert self.sum_tds.min() >= 0
            assert self.cum_tds.min() >= 0

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        td_errors = self.td_errors[played_mask]
        weight = (td_errors * (1 - (subsequent_tds[played_mask] / self._max_sum_follow_td))) ** self._alpha

        self.sampling_weights[played_mask] = weight

    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability_discount = subsequent_tds / self._max_sum_follow_td 
        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = (self.td_errors[ep_done_row_idxes, ep_done_col_idxes] * (1 - reliability_discount)) ** self._alpha



class CustomPrioritizedReplayBufferCumSum5(CustomPrioritizedReplayBufferCumSum):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = 1.,
            check_frequency: int = 100_000,
    ):
        
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        self._max_sum_follow_td = 1.
        self.reward_ratios = np.zeros((self.buffer_size, self.n_envs))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        self.reward_ratios[self.pos] = 1.
        self.sum_tds[self.pos] = self._max_sum_follow_td
        super().add(obs, next_obs, action, reward, done, infos)

    # def add(
    #     self,
    #     obs: np.ndarray,
    #     next_obs: np.ndarray,
    #     action: np.ndarray,
    #     reward: np.ndarray,
    #     done: np.ndarray,
    #     infos: List[Dict[str, Any]],
    # ) -> None:
        
    #     if self.debug_mode:
    #         assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

    #     self.td_errors[self.pos] = self._max_td
    #     self.episodes[self.pos] = self._current_episode
    #     self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
    #     self.timesteps[self.pos] = self._current_timestep
    #     self.reward_ratios[self.pos] = 1.
        
    #     self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + self._max_td
    #     self.sum_tds[self.pos] = self._max_sum_follow_td

    #     self.sampling_weights[self.pos] = self._max_td ** self._alpha # Initiate with max priority & reliability of 1

    #     # Compute actual sampling weights once episode is done
    #     if done.any():
    #         ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
    #         self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
    #         self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

    #     # Update tracking variables
    #     self._current_timestep = 1 + (self._current_timestep * ~done)
    #     self._current_episode += self.n_envs * done
    #     self.last_done = done

    #     super().add(obs, next_obs, action, reward, done, infos)
    
    def update_priorities(self, idxes, priorities, reward_ratios):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert np.min(reward_ratios) >= 0
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        priorities = priorities[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]
        reward_ratios = reward_ratios[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - priorities

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Update reward ratios
        self.reward_ratios[row_idxes, col_idxes] = reward_ratios

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        self.update_sampling_weights(played_and_change_mask)
        self.full_check()

    # def update_priorities(self, idxes, priorities, reward_ratios):
        # assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        # row_idxes, col_idxes = idxes

        # if self.debug_mode:
        #     assert np.min(reward_ratios) >= 0
        #     assert len(col_idxes) == len(priorities)
        #     assert len(row_idxes) == len(priorities)
        #     assert np.min(priorities) > 0, f"{np.min(priorities)=}"
        #     assert np.min(idxes) >= 0
        #     assert np.max(idxes) < len(self.observations)

        #     ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
        #     if ep_done_mask.any():
        #         assert np.all(self.cum_tds[ep_done_mask] >= 0)
        #         assert np.all(self.sum_tds[ep_done_mask] >= 0)
        #         assert np.all(self.td_errors[ep_done_mask] >= 0)

        # # Use set to remove duplicates faster
        # unique_vals, unique_indices = np.unique(np.ravel_multi_index(idxes, self.td_errors.shape), return_index=True)
        # row_idxes = row_idxes[unique_indices]
        # col_idxes = col_idxes[unique_indices]
        # priorities = priorities[unique_indices]
        # reward_ratios = reward_ratios[unique_indices]

        # episodes_to_update = self.episodes[row_idxes, col_idxes]
        # timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        # prev_td_errors = self.td_errors[row_idxes, col_idxes]
        # deltas = prev_td_errors - priorities

        # self.td_errors[row_idxes, col_idxes] = priorities
        # self.reward_ratios[row_idxes, col_idxes] = reward_ratios

        # not_current_ep = episodes_to_update != self._current_episode

        # if np.any(not_current_ep):
        #     played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        #     # Only process if any valid entries
        #     if np.any(played_and_change_mask):
        #         ep_masked = self.episodes[played_and_change_mask]
        #         ts_masked = self.timesteps[played_and_change_mask]

        #         # Compute episode match mask more efficiently with broadcasting
        #         sum_mask = ep_masked[:, None] == episodes_to_update[None, :]
        #         sum_deltas = deltas * not_current_ep

        #         self.sum_tds[played_and_change_mask] -= sum_mask @ sum_deltas

        #         cum_mask = sum_mask & (ts_masked[:, None] >= timesteps_to_update[None, :])
        #         self.cum_tds[played_and_change_mask] -= cum_mask @ deltas

        # self._max_td = max(self._max_td, np.max(priorities))
        # self.update_sampling_weights(played_and_change_mask)

    def update_sampling_weights(self, played_mask=None): # 3.4
                
        if self.debug_mode:
            assert self.td_errors.min() >= 0
            # assert self.sum_tds.min() >= 0, f"{self.sum_tds.min()=}"
            assert self.cum_tds.min() >= 0
            assert self.reward_ratios.min() >= 0

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        is_current_episode = np.isin(self.episodes[played_mask], self._current_episode, kind="table")

        td_errors = self.td_errors[played_mask]
        reward_ratios = self.reward_ratios[played_mask]

        weight_current_episode = td_errors ** self._alpha

        reliability = (1 - (subsequent_tds[played_mask] / self._max_sum_follow_td))
        weight_other_episodes = ((td_errors * reliability * (1-reward_ratios)) + (td_errors * reward_ratios)) ** self._alpha

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = np.where(is_current_episode, weight_current_episode, weight_other_episodes)

            
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability = 1 - (subsequent_tds / self._max_sum_follow_td)
        td_errors = self.td_errors[ep_done_row_idxes, ep_done_col_idxes]
        reward_ratios = self.reward_ratios[ep_done_row_idxes, ep_done_col_idxes]

        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = ((td_errors * reliability * (1-reward_ratios)) + (td_errors * reward_ratios)) ** self._alpha


class CustomPrioritizedReplayBufferCumSum6(CustomPrioritizedReplayBufferCumSum):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = 1.,
            check_frequency: int = 100_000,
    ):
        
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        self._max_sum_follow_td = 1.
        self.reward_ratios = np.zeros((self.buffer_size, self.n_envs))

        print(f"{self.handle_timeout_termination=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        if self.debug_mode:
            assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

        self.td_errors[self.pos] = self._max_td
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep
        self.reward_ratios[self.pos] = 1.
        
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + self._max_td
        self.sum_tds[self.pos] = self._max_sum_follow_td

        self.sampling_weights[self.pos] = self._max_td ** self._alpha # Initiate with max priority & reliability of 1

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)
    
    def update_priorities(self, idxes, priorities, reward_ratios):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert np.min(reward_ratios) >= 0
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        priorities = priorities[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]
        reward_ratios = reward_ratios[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - priorities

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update)


        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Update reward ratios
        self.reward_ratios[row_idxes, col_idxes] = reward_ratios

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        # self.full_check()
        self.update_sampling_weights(played_and_change_mask)

    def update_sampling_weights(self, played_mask=None): # 3.4
        
        is_current_episode = np.isin(self.episodes[played_mask], self._current_episode, kind="table")

        if self.debug_mode:

            not_current_episode_mask = ~is_current_episode
            assert self.td_errors[not_current_episode_mask].min() >= 0
            assert self.sum_tds[not_current_episode_mask].min() >= 0
            assert self.cum_tds[not_current_episode_mask].min() >= 0
            assert self.reward_ratios[not_current_episode_mask].min() >= 0

        subsequent_tds = self.sum_tds - self.cum_tds
        # self._max_sum_follow_td = np.max(subsequent_tds)

        td_errors = self.td_errors[played_mask]
        reward_ratios = self.reward_ratios[played_mask]

        weight_current_episode = td_errors ** self._alpha

        reliability = (1 - (subsequent_tds[played_mask] / self.sum_tds[played_mask]))
        weight_other_episodes = ((td_errors * reliability * (1-reward_ratios)) + (td_errors * reward_ratios)) ** self._alpha

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = np.where(is_current_episode, weight_current_episode, weight_other_episodes)
        
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        # self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability = 1 - (subsequent_tds / self.sum_tds[ep_done_row_idxes, ep_done_col_idxes])
        td_errors = self.td_errors[ep_done_row_idxes, ep_done_col_idxes]
        reward_ratios = self.reward_ratios[ep_done_row_idxes, ep_done_col_idxes]

        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = ((td_errors * reliability * (1-reward_ratios)) + (td_errors * reward_ratios)) ** self._alpha


class CustomPrioritizedReplayBufferCumSum7(CustomPrioritizedReplayBufferCumSum):
    """
    with alpha2 and alpha1, seperately tuneable
    with reliability boost for immediate reward
    with reliability estimation for ongoing runs
    with reliability spanning 0 to 1 for all runs
    """

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = .4,
            check_frequency: int = 100_000,
            use_reward_ratios:bool = False,
    ):
        
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        
        self.use_reward_ratios = use_reward_ratios

        self._max_sum_follow_td = 1.
        self.reward_ratios = np.zeros((self.buffer_size, self.n_envs))

        print(f"{self.handle_timeout_termination=}")
        print(f"{self.use_reward_ratios=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        self.reward_ratios[self.pos] = 1.
        self.sum_tds[self.pos] = self._max_sum_follow_td
        super().add(obs, next_obs, action, reward, done, infos)

    def update_priorities(self, idxes, priorities, reward_ratios):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert np.min(reward_ratios) >= 0
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        priorities = priorities[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]
        reward_ratios = reward_ratios[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - priorities

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Update reward ratios
        self.reward_ratios[row_idxes, col_idxes] = reward_ratios

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        sum_deltas = deltas * (episodes_to_update != self._current_episode)
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, sum_deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        self.full_check()
        self.update_sampling_weights(played_and_change_mask)


    def update_sampling_weights(self, played_mask=None): # 3.4

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        td_errors = self.td_errors[played_mask]
        reward_ratios = self.reward_ratios[played_mask]
        reliability = (1 - (subsequent_tds[played_mask] / self.sum_tds[played_mask]))

        if self.use_reward_ratios:
            new_weights = (td_errors**self._alpha * reliability**self._alpha2 * (1-reward_ratios)) \
                        + (td_errors ** self._alpha * reward_ratios)
        else:
            new_weights = td_errors**self._alpha * reliability**self._alpha2

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = new_weights
        
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability = 1 - (subsequent_tds / self.sum_tds[ep_done_row_idxes, ep_done_col_idxes])
        td_errors = self.td_errors[ep_done_row_idxes, ep_done_col_idxes]

        if self.use_reward_ratios:
            reward_ratios = self.reward_ratios[ep_done_row_idxes, ep_done_col_idxes]
            self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = (td_errors**self._alpha * reliability**self._alpha2 * (1-reward_ratios)) \
                                                                        + (td_errors**self._alpha * reward_ratios)
        else:
            self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = td_errors**self._alpha * reliability**self._alpha2


class CustomPrioritizedReplayBufferCumSum8(CustomPrioritizedReplayBufferCumSum):
    """
    with alpha2 and alpha1, seperately tuneable
    with reliability boost for immediate reward
    with reliability estimation for ongoing runs
    """

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = .4,
            check_frequency: int = 100_000,
            use_reward_ratios:bool = False,
    ):
        
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        
        self.use_reward_ratios = use_reward_ratios

        self._max_sum_follow_td = 1.
        self.reward_ratios = np.zeros((self.buffer_size, self.n_envs))

        print(f"{self.handle_timeout_termination=}")
        print(f"{self.use_reward_ratios=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        self.reward_ratios[self.pos] = 1.
        self.sum_tds[self.pos] = self._max_sum_follow_td
        super().add(obs, next_obs, action, reward, done, infos)

    def update_priorities(self, idxes, priorities, reward_ratios):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert np.min(reward_ratios) >= 0
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        priorities = priorities[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]
        reward_ratios = reward_ratios[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - priorities

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Update reward ratios
        self.reward_ratios[row_idxes, col_idxes] = reward_ratios

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        sum_deltas = deltas * (episodes_to_update != self._current_episode)
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, sum_deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        self.full_check()
        self.update_sampling_weights(played_and_change_mask)


    def update_sampling_weights(self, played_mask=None): # 3.4

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        td_errors = self.td_errors[played_mask]
        reward_ratios = self.reward_ratios[played_mask]
        reliability = (1 - (subsequent_tds[played_mask] / self.sum_tds[played_mask]))

        if self.use_reward_ratios:
            new_weights = (td_errors**self._alpha * reliability**self._alpha2 * (1-reward_ratios)) \
                        + (td_errors ** self._alpha * reward_ratios)
        else:
            new_weights = td_errors**self._alpha * reliability**self._alpha2

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = new_weights
        
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability = 1 - (subsequent_tds / self.sum_tds[ep_done_row_idxes, ep_done_col_idxes])
        td_errors = self.td_errors[ep_done_row_idxes, ep_done_col_idxes]

        if self.use_reward_ratios:
            reward_ratios = self.reward_ratios[ep_done_row_idxes, ep_done_col_idxes]
            self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = (td_errors**self._alpha * reliability**self._alpha2 * (1-reward_ratios)) \
                                                                        + (td_errors**self._alpha * reward_ratios)
        else:
            self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = td_errors**self._alpha * reliability**self._alpha2


class CustomPropagatingPrioritizedReplayBuffer(CustomPrioritizedReplayBufferCumSum):

    def update_priorities(self, idxes, priorities, learning_rate = 0.0001):
        row_idxes, col_idxes = idxes

        deltas = self.td_errors[row_idxes, col_idxes] - priorities
        self.td_errors[row_idxes, col_idxes] - deltas

        prop_mask = (self.timesteps[row_idxes, col_idxes] != 1.) & (deltas > 0)
        prop_row_idxes = row_idxes[prop_mask] - 1
        prop_row_idxes[prop_row_idxes==-1] = self.buffer_size-1

        prop_col_idxes = col_idxes[prop_mask]
        prop_deltas = deltas[prop_mask]

        self.td_errors[prop_row_idxes, prop_col_idxes] = self.td_errors[prop_row_idxes, prop_col_idxes] + prop_deltas * learning_rate

        self.update_sampling_weights()

    def update_sampling_weights(self):
        self.sampling_weights = self.td_errors ** self._alpha

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        
        if self.debug_mode:
            assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

        self.td_errors[self.pos] = self._max_td
        self.sampling_weights[self.pos] = self._max_td ** self._alpha

        super().add(obs, next_obs, action, reward, done, infos)


class CustomPropagatingPrioritizedReplayBufferCumSum(CustomPrioritizedReplayBufferCumSum):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .6,
            log_path: str = "",
            debug_mode: bool = False,
            alpha2: float = 1.,
            check_frequency: int = 100_000,
    ):
        
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, 
            handle_timeout_termination=handle_timeout_termination, alpha=alpha,log_path = log_path, debug_mode=debug_mode, 
            alpha2=alpha2, check_frequency=check_frequency,
            )
        self._max_sum_follow_td = 1.
        self.reward_ratios = np.zeros((self.buffer_size, self.n_envs))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        if self.debug_mode:
            assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"

        self.td_errors[self.pos] = self._max_td
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep
        self.reward_ratios[self.pos] = 1.
        
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[[self.pos]-1]) + self._max_td
        self.sum_tds[self.pos] = self._max_sum_follow_td

        self.sampling_weights[self.pos] = self._max_td ** self._alpha # Initiate with max priority & reliability of 1

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)
    
    def update_priorities(self, idxes, priorities, reward_ratios, learning_rate):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert np.min(reward_ratios) >= 0
            assert len(col_idxes) == len(priorities)    
            assert len(row_idxes) == len(priorities)
            assert np.min(priorities) > 0, f"{np.min(priorities)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        priorities = priorities[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]
        reward_ratios = reward_ratios[unique_idx]

        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - priorities

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        prop_mask = (self.timesteps[row_idxes, col_idxes] != 1.) & (deltas > 0)
        prop_row_idxes = row_idxes[prop_mask] - 1
        prop_row_idxes[prop_row_idxes==-1] = self.buffer_size-1

        prop_col_idxes = col_idxes[prop_mask]
        prop_deltas = deltas[prop_mask]

        self.td_errors[prop_row_idxes, prop_col_idxes] = self.td_errors[prop_row_idxes, prop_col_idxes] + prop_deltas * learning_rate

        # Update reward ratios
        self.reward_ratios[row_idxes, col_idxes] = reward_ratios

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(priorities))

        # self.full_check()
        self.update_sampling_weights(played_and_change_mask)

    def update_sampling_weights(self, played_mask=None): # 3.4

        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_follow_td = np.max(subsequent_tds)

        is_current_episode = np.isin(self.episodes[played_mask], self._current_episode, kind="table")

        td_errors = self.td_errors[played_mask]
        reward_ratios = self.reward_ratios[played_mask]

        weight_current_episode = td_errors ** self._alpha

        reliability = (1 - (subsequent_tds[played_mask] / self._max_sum_follow_td))
        weight_other_episodes = ((td_errors * reliability * (1-reward_ratios)) + (td_errors * reward_ratios)) ** self._alpha

        # Use np.where only once for efficiency
        self.sampling_weights[played_mask] = np.where(is_current_episode, weight_current_episode, weight_other_episodes)
        
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]
        self._max_sum_follow_td = max(self._max_sum_follow_td, np.max(subsequent_tds))

        reliability = 1 - (subsequent_tds / self._max_sum_follow_td)
        td_errors = self.td_errors[ep_done_row_idxes, ep_done_col_idxes]
        reward_ratios = self.reward_ratios[ep_done_row_idxes, ep_done_col_idxes]

        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = ((td_errors * reliability * (1-reward_ratios)) + (td_errors * reward_ratios)) ** self._alpha
