import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import logging
import numpy as np

class ReaPER(ReplayBuffer):

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
            alpha2: float = .6,
            check_frequency: int = 100_000,
            use_reward_ratios: bool = False,
            max_sum_normalization: bool = False,
            conservative_initial_reliabilities: bool = True,
            update_sums: bool = True,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

        assert n_envs == 1, "Multiple environments currently not supported."

        self.debug_mode = debug_mode
        self._alpha = np.float64(alpha)
        self._alpha2 = np.float64(alpha2)

        self.check_counter = 0
        self.check_frequency = check_frequency

        self.use_reward_ratios = use_reward_ratios
        self.max_sum_normalization = max_sum_normalization
        self.conservative_initial_reliabilities = conservative_initial_reliabilities
        self.update_sums = update_sums

        self._max_sum_td = 1.
        self.reward_ratios = np.zeros((self.buffer_size, self.n_envs))

        # Setup sampling array
        self.sample_arange = np.arange(self.buffer_size * self.n_envs)

        # Setup episode storage
        self._current_episode = np.arange(1, self.n_envs+1, dtype=int)
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=int)
        self.episodes_played = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # Setup td error, td sum and td cumsum storage
        self._max_td = 1.
        self.max_td_changed = False
        self.last_done = np.zeros(self.n_envs,dtype=np.bool_)
        self._cum_td = np.zeros(self.n_envs,)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        # self.init_sum_tds = np.zeros((self.buffer_size, self.n_envs))
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

        # Set preliminary sums and reward ratios
        self.reward_ratios[self.pos] = 1.
        self.sum_tds[self.pos] = self._max_sum_td
        # self.init_sum_tds[self.pos] = self._max_sum_td
        
        # Add transition information
        self.td_errors[self.pos] = self._max_td
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + self._max_td

        # Initiate sampling weights at max td
        self.sampling_weights[self.pos] = self.td_errors[self.pos] ** self._alpha

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            # self.init_sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self._max_td * self._current_timestep #[self.pos][ep_done_col_idxes]
            self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)

    def sample_naive(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):

        assert beta > 0
        assert self.n_envs == 1
        
        if self.debug_mode:
            assert self.sampling_weights.min() >= 0

        num_transitions_gathered = (self.pos if not self.full else self.buffer_size) * self.n_envs
        sampling_probas = self.sampling_weights.flatten() / self.sampling_weights.sum()
        row_idxes = np.random.choice(self.sample_arange, p=sampling_probas, size=batch_size)
        col_idxes = np.zeros(shape=(batch_size,), dtype=np.int64)

        # Encode
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Get importance sampling weights
        IS_weights = []
        p_min = sampling_probas[:num_transitions_gathered].min()
        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        sampling_probas_of_batch = sampling_probas[row_idxes]
        IS_weights = (sampling_probas_of_batch * num_transitions_gathered) ** (-beta) / max_weight

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)
    
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

        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Get importance sampling weights
        sampling_weight_sum = row_weights[-1]
        sampling_probas_of_batch = self.sampling_weights[row_idxes, col_idxes] / sampling_weight_sum
        p_min = self.sampling_weights[:num_transitions_gathered].min() / sampling_weight_sum
        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        IS_weights = (sampling_probas_of_batch * num_transitions_gathered) ** (-beta) / max_weight

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)
    
    def update_priorities(self, idxes, new_td_errors, reward_ratios):

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        if self.debug_mode:
            assert np.min(reward_ratios) >= 0
            assert len(col_idxes) == len(new_td_errors)    
            assert len(row_idxes) == len(new_td_errors)
            assert np.min(new_td_errors) > 0, f"{np.min(new_td_errors)=}"
            assert np.min(idxes) >= 0
            assert np.max(idxes) < len(self.observations)

            ep_done_mask = self.episodes_played & (self.episodes != self._current_episode)
            if ep_done_mask.any():
                assert np.min(self.cum_tds[ep_done_mask] >= 0)
                assert np.min(self.sum_tds[ep_done_mask] >= 0)
                assert np.min(self.td_errors[ep_done_mask] >= 0)

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)

        if not self.conservative_initial_reliabilities: # Leave sampling weights at max until episode is finished if conservative initialization is off
            unique_idx = unique_idx[self.episodes[:, 0][unique_idx] != self._current_episode]

        # print(f"{unique_idx=}")
        new_td_errors = new_td_errors[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]
        reward_ratios = reward_ratios[unique_idx]

        # Mask relevant variables
        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - new_td_errors

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update, kind="table")

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
        if not self.max_td_changed:
            self._max_td = np.max(new_td_errors)
            self.max_td_changed = True

        self._max_td = max(self._max_td, np.max(new_td_errors))

        self.update_sampling_weights(played_and_change_mask)
        self.full_check()
        # print("---")

    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        
        episode_mask = ep_done_row_idxes, ep_done_col_idxes

        # Mask relevant variables
        td_errors = self.td_errors[episode_mask]
        sum_tds = self.sum_tds[episode_mask]
        cum_tds = self.cum_tds[episode_mask]

        subsequent_tds = sum_tds - cum_tds
        self._max_sum_td = max(self._max_sum_td, np.max(sum_tds))

        if self.max_sum_normalization:
            reliability = 1 - (subsequent_tds / self._max_sum_td)
        elif not self.max_sum_normalization:
            reliability = 1 - (subsequent_tds / sum_tds)

        regularized_td = td_errors**self._alpha
        if self.use_reward_ratios:
            reward_ratios = self.reward_ratios[ep_done_row_idxes, ep_done_col_idxes]
            self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = (regularized_td * reliability**self._alpha2 * (1-reward_ratios)) \
                                                                        + (regularized_td * reward_ratios)
        else:
            self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = regularized_td * reliability**self._alpha2


    def update_sampling_weights(self, played_mask=None):

        # Calculate subsequent TDs and update maximum subsequent TD
        self._max_sum_td = np.max(self.sum_tds)

        # Mask relevant variables
        sum_tds = self.sum_tds[played_mask]
        cum_tds = self.cum_tds[played_mask]
        td_errors = self.td_errors[played_mask]
        reward_ratios = self.reward_ratios[played_mask]

        # Get subsequent errors
        subsequent_tds = sum_tds - cum_tds

        # Compute reliability
        if self.max_sum_normalization:
            reliability = 1 - (subsequent_tds / self._max_sum_td)
        elif not self.max_sum_normalization:
            reliability = 1 - (subsequent_tds / sum_tds)

        # Update sampling weights
        regularized_td = td_errors**self._alpha
        if self.use_reward_ratios:
            new_weights = regularized_td * (reliability ** self._alpha2) * (1-reward_ratios) \
                        + regularized_td * reward_ratios
        else:
            new_weights = regularized_td * reliability ** self._alpha2

        self.sampling_weights[played_mask] = new_weights

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

class R_UNI(ReaPER):

    def sample(self, batch_size: int, beta: float, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        upper_bound = self.buffer_size if self.full else self.pos
        
        row_idxes = np.random.randint(0, upper_bound, size=batch_size)
        col_idxes = np.zeros_like(row_idxes) # col_idxes = np.random.randint(0, self.n_envs, size=batch_size)
        batch_idxes = row_idxes, col_idxes

        # Mask relevant variables
        sum_tds = self.sum_tds[batch_idxes]
        cum_tds = self.cum_tds[batch_idxes]
        subsequent_tds = sum_tds - cum_tds
        

        if self.max_sum_normalization:
            reliability = (1 - (subsequent_tds / self._max_sum_td)) ** self._alpha2
        else:
            reliability = (1 - (subsequent_tds / sum_tds)) ** self._alpha2


        # reliability = 2 ** (2 * reliability - 1)
        # reliability * 2
        # reliability = 2. ** (2 * reliability - 1) # Ensure that loss can at most be doubled and at least be halfed

        # print(reliability.min(), reliability.max(), reliability.mean())
        # reliability = reliability**2 + (0.5 * reliability) + 0.5 # Ensure that loss can at most be doubled and at least be halfed
        # reliability = self._alpha2 ** (2 * reliability - 1)
        # print(reliability.min(), reliability.max(), reliability.mean())
        # print("---")

        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), reliability, (row_idxes, col_idxes)
    
    def calculate_sampling_weights_for_finished_runs(self, ep_done_row_idxes, ep_done_col_idxes):
        self._max_sum_td = max(self._max_sum_td, np.max(self.sum_tds[ep_done_row_idxes, ep_done_col_idxes]))

    def update_sampling_weights(self, played_mask):
        self._max_sum_td = np.max(self.sum_tds)

        # if self.ep_to_track != (self._current_episode - 1):

            # ep_to_track = (self._current_episode - 1)
            # self.ep_to_track = ep_to_track
            # ep_mask = self.episodes == ep_to_track
            # print(f"{ep_to_track=}")
            # print(f"{self._max_td=}")
            # print(f"{self.timesteps[ep_mask]=}")
            # print(f"{self.cum_tds[ep_mask]=}")
            # print(f"{self.sum_tds[ep_mask]=}")
            # # print(f"{self.init_sum_tds[ep_mask]=}")

            # sum_tds = self.sum_tds[ep_mask]
            # cum_tds = self.cum_tds[ep_mask]
            # subsequent_tds = sum_tds - cum_tds
            # # init_sum_tds = self.init_sum_tds[ep_mask]

            # # raw_reliability = (1 - (subsequent_tds / init_sum_tds))
            # # reliability = raw_reliability ** self._alpha2
            # print(f"{subsequent_tds=}")
            # print("\n----\n")

            # self.ep_to_track = ep_to_track