import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.segment_tree import SumSegmentTree, MinSegmentTree
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any, Tuple
from gymnasium import spaces
import torch as th
from sklearn.metrics import r2_score
from scipy.special import softmax
import warnings
import copy
from utils.storage import get_export_path
import logging
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import warnings
import logging
import matplotlib.pyplot as plt


# archived on 20241203
class CustomPrioritizedReplayBufferCumSum(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            alpha: float = .6,
            log_path: str = "",
            custom_buffer_size = None,
            use_importance_sampling: bool = True,
            disable_within_episode_weights: bool = False,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)
        
        # Overwrite buffer size with custom value if given
        if custom_buffer_size is not None:
            self.buffer_size = custom_buffer_size

        self._alpha = alpha
        self.use_importance_sampling = use_importance_sampling
        self.disable_within_episode_weights = disable_within_episode_weights

        # Setup sampling array
        self.sample_arange = np.arange(self.buffer_size * self.n_envs)

        # Setup episode storage
        self._current_episode = np.arange(1, n_envs+1)
        self.episodes = np.zeros((self.buffer_size, self.n_envs))
        
        # Setup td error, td sum and td cumsum storage
        self._max_td = 1
        self._last_cum_td = np.zeros(self.n_envs, )
        self._cum_td = np.zeros(n_envs,)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.td_errors = np.zeros((self.buffer_size, self.n_envs))

        # Setup timestep storage
        self._current_timestep = np.ones(n_envs,)
        self.timesteps = np.zeros((self.buffer_size, self.n_envs))

        # Setup logger
        self.log_path = log_path
        self.logger = logging.getLogger(__name__)
        # logging.basicConfig(filename=self.log_path + 'replay_value_prediction_model.log', encoding='utf-8', level=logging.DEBUG)
        self.logger.info("Weight prediction model summary\n---")

        print(f"{self._alpha=}")
        print(f"{self.disable_within_episode_weights=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """

        assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"
        idx = self.pos

        super().add(obs, next_obs, action, reward, done, infos)        
        
        self.td_errors[idx] = self._max_td
        self.episodes[idx] = self._current_episode
        self.timesteps[idx] = self._current_timestep
        self.cum_tds[idx] = self._last_cum_td + self._max_td
        self.sum_tds[self.episodes==self._current_episode] = self.cum_tds[idx]

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self._last_cum_td = self.cum_tds[idx] * ~done

    def get_within_episode_weights(self):
        """
        calculates weights for each timestep quantifying the reliability of TD errors based on the amount and size of subsequent TD errors
        """
        weights = self.cum_tds / self.sum_tds
        weights = np.where(self.episodes==0, 0, weights)
        return weights
    
    def sample(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        within_episode_weights = self.get_within_episode_weights()

        if self.disable_within_episode_weights:
            weighted_tdes = (self.td_errors) ** self._alpha
        else:
            weighted_tdes = (self.td_errors * within_episode_weights) ** self._alpha

        sampling_probs = weighted_tdes / weighted_tdes.sum()
        flat_sampling_probs = sampling_probs.reshape(-1)
        non_zero_mask = flat_sampling_probs!=0
        sample_idxes = np.random.choice(self.sample_arange[non_zero_mask], batch_size, p=flat_sampling_probs[non_zero_mask])
        
        row_idxes = sample_idxes//self.n_envs
        col_idxes = sample_idxes%self.n_envs
        
        sampling_probas_of_sample = flat_sampling_probs[sample_idxes]
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # Get importance sampling weights
        num_transitions_gathered = (self.buffer_size if self.full else self.pos) * self.n_envs
        p_min = sampling_probs[:num_transitions_gathered].min()
        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        IS_weights = (sampling_probas_of_sample * num_transitions_gathered) ** (-beta) / max_weight

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)
    
    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        if len(unique_idx) < len(priorities):
            priorities = priorities[unique_idx]
            row_idxes = row_idxes[unique_idx]
            col_idxes = col_idxes[unique_idx]
            warnings.warn("The same transition was sampled multiple times.")

        assert len(col_idxes) == len(priorities)
        assert len(row_idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.observations)
        assert (self.td_errors < 0).sum() == 0
        assert (self.cum_tds < 0).sum() == 0
        assert (self.sum_tds < 0).sum() == 0

        current_tds = self.td_errors[row_idxes, col_idxes]
        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = current_tds-priorities

        # Update TD Errors
        self.td_errors[row_idxes, col_idxes] = priorities

        # Update cumsum & sum
        for (episode_to_update, timestep_to_update, delta) in zip(episodes_to_update, timesteps_to_update, deltas):
            self.sum_tds[self.episodes == episode_to_update] -= delta
            self.cum_tds[(self.episodes == episode_to_update) & (self.timesteps >= timestep_to_update)] -= delta

        self._max_td = max(self._max_td, np.max(priorities))

    def plot_sampling_distributions(self, tds_split_by_eps, weights_by_eps):

        assert self.n_envs==1, "Currently incompatible with multiple environments"

        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15, 15))
        all_tde_bin_means, all_weight_bin_means = [], []

        for episode_idx, (tdes, weights) in enumerate(zip(tds_split_by_eps, weights_by_eps)):
            
            if not tdes.sum() == 0:
                run_progress = np.arange(len(tdes)) / len(tdes)

                num_bins = 10
                bins = np.linspace(0, 1, num_bins)
                digitized = np.digitize(run_progress, bins)

                tde_bin_means = [tdes[digitized == i].mean() for i in range(1, len(bins))]
                weights_bin_means = [weights[digitized == i].mean() for i in range(1, len(bins))]

                all_weight_bin_means.append(weights_bin_means)
                all_tde_bin_means.append(tde_bin_means)
        
        weight_dist = np.nanmean(np.array(all_weight_bin_means), axis=0)
        tde_dist = np.nanmean(np.array(all_tde_bin_means), axis=0)
        run_progress = np.arange(0, len(weight_dist))

        axs[0].bar(run_progress, tde_dist)
        axs[0].set_title("TDEs")
        axs[1].bar(run_progress, weight_dist)
        axs[1].set_title("Weights")
            
        # axs[episode_idx,0].bar(step_idx, tdes)
        # axs[episode_idx,0].set_title("TDEs")

        # axs[episode_idx,1].bar(step_idx, tdes ** self._alpha)
        # axs[episode_idx,1].set_title("TDEs ** Alpha")

        # axs[episode_idx,2].bar(step_idx, weights)
        # axs[episode_idx,2].set_title("Weights")

        # axs[episode_idx,3].bar(step_idx, tdes*weights)
        # axs[episode_idx,3].set_title("TDEs x Weights")

        # axs[episode_idx,4].bar(step_idx, (tdes*weights) ** self._alpha)
        # axs[episode_idx,4].set_title("TDEs x Weights ** Alpha")

        print("Saving...")
        plt.savefig("./results/_latest_tde_plots/" + f"{self._current_episode}ep_plots.png")

# ---
class CustomPrioritizedReplayBufferCumSum(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            alpha: float = .6,
            log_path: str = "",
            custom_buffer_size = None,
            use_importance_sampling: bool = True,
            continuity_batch_size_prop: float = 0.0,
            disable_within_episode_weights: bool = False,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)
        
        # Overwrite buffer size with custom value
        if custom_buffer_size is not None:
            self.buffer_size = custom_buffer_size

        self.use_importance_sampling = use_importance_sampling
        self._alpha = alpha
        self.td_errors = np.zeros(self.buffer_size,)
        self._max_td = 1
        self.continuity_batch_idxes = np.array([], dtype=np.bool_)
        self.continuity_batch_size_prop = continuity_batch_size_prop
        self.disable_within_episode_weights = disable_within_episode_weights

        self.episodes = np.zeros(self.buffer_size,)
        self._current_episode = 1

        self.log_path = log_path
        self.logger = logging.getLogger(__name__)
        # logging.basicConfig(filename=self.log_path + 'replay_value_prediction_model.log', encoding='utf-8', level=logging.DEBUG)
        self.logger.info("Weight prediction model summary\n---")

        print(f"{self._alpha=}")
        print(f"{self.disable_within_episode_weights=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self.pos

        super().add(obs, next_obs, action, reward, done, infos)

        self.td_errors[idx] = self._max_td
        self.episodes[idx] = self._current_episode

        if done:
            self._current_episode += 1

    def get_within_episode_weights(self, td_errors, episodes):
        """
        calculates weights for each timestep quantifying the reliability of TD errors based on the amount and size of subsequent TD errors
        """
        episode_spillover = episodes[0] == episodes[-1]

        if episode_spillover:
            spillover_ep = episodes[-1]
            timesteps_from_spillover_ep = episodes == spillover_ep
            first_timestep_of_spillover_ep = -np.argmax(~timesteps_from_spillover_ep[::-1])

            # If there is spillover, take the timesteps from the end of the buffer and put them at the start to enable correct cumsums
            episodes = np.concatenate([episodes[first_timestep_of_spillover_ep:], episodes[:first_timestep_of_spillover_ep]])
            td_errors = np.concatenate([td_errors[first_timestep_of_spillover_ep:], td_errors[:first_timestep_of_spillover_ep]])

        # Get sub-arrays for each episode, calculate within-episode cumsums and flatten
        _, idxes = np.unique(episodes, return_index=True)
        tds_split_by_eps = np.split(td_errors, sorted(idxes))[1:]
        weights_by_eps = [np.cumsum(tds_split_by_ep)/np.sum(tds_split_by_ep) for tds_split_by_ep in tds_split_by_eps]
        weights = np.concatenate(weights_by_eps)

        # if self._current_episode%500 == 0:
        #    self.plot_sampling_distributions(tds_split_by_eps, weights_by_eps)

        if episode_spillover:
            # Restore original order
            weights = np.concatenate([weights[-first_timestep_of_spillover_ep:], weights[:-first_timestep_of_spillover_ep]])

        weights = np.nan_to_num(weights)

        return weights
    
    def sample(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        within_episode_weights = self.get_within_episode_weights(self.td_errors, self.episodes) 

        if self.disable_within_episode_weights:
            weighted_tdes = (self.td_errors) ** self._alpha
        else:
            weighted_tdes = (self.td_errors * within_episode_weights) ** self._alpha
        
        sampling_probs = weighted_tdes / weighted_tdes.sum()

        to_sample = batch_size - len(self.continuity_batch_idxes)
        newly_sampled_sample_idxes = np.random.choice(a=np.arange(self.buffer_size), size=to_sample, p=sampling_probs)
        sample_idxes = np.concatenate([self.continuity_batch_idxes, newly_sampled_sample_idxes])
        
        sampling_probas_of_sample = sampling_probs[sample_idxes]

        encoded_sample = super()._get_samples(sample_idxes, env=env)

        # Get importance sampling weights
        if self.use_importance_sampling:
            num_transitions_gathered = (self.buffer_size if self.full else self.pos) * self.n_envs
            p_min = sampling_probs[:num_transitions_gathered].min()
            max_weight = (p_min * num_transitions_gathered) ** (-beta)
            IS_weights = (sampling_probas_of_sample * num_transitions_gathered) ** (-beta) / max_weight

        else:
            IS_weights = np.ones_like(sample_idxes)

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, sample_idxes
    
    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        # assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.observations)

        if self.continuity_batch_size_prop > 0:
            prio_diff = self.td_errors[idxes] - priorities
            continuity_batch_size = int(len(idxes) * self.continuity_batch_size_prop)
            order_idxes = np.argsort(prio_diff)
            idxes_ranked_by_diff = idxes[order_idxes]
            highest_ranked_idxes = idxes_ranked_by_diff[-continuity_batch_size:]
            self.continuity_batch_idxes = highest_ranked_idxes - 1

        self.td_errors[idxes] = priorities
        self._max_td = max(self._max_td, np.max(priorities))

    def plot_sampling_distributions(self, tds_split_by_eps, weights_by_eps):

        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15, 15))
        all_tde_bin_means, all_weight_bin_means = [], []

        for episode_idx, (tdes, weights) in enumerate(zip(tds_split_by_eps, weights_by_eps)):
            
            if not tdes.sum() == 0:
                run_progress = np.arange(len(tdes)) / len(tdes)

                num_bins = 10
                bins = np.linspace(0, 1, num_bins)
                digitized = np.digitize(run_progress, bins)

                tde_bin_means = [tdes[digitized == i].mean() for i in range(1, len(bins))]
                weights_bin_means = [weights[digitized == i].mean() for i in range(1, len(bins))]

                all_weight_bin_means.append(weights_bin_means)
                all_tde_bin_means.append(tde_bin_means)

        
        weight_dist = np.nanmean(np.array(all_weight_bin_means), axis=0)
        tde_dist = np.nanmean(np.array(all_tde_bin_means), axis=0)
        run_progress = np.arange(0, len(weight_dist))

        axs[0].bar(run_progress, tde_dist)
        axs[0].set_title("TDEs")
        axs[1].bar(run_progress, weight_dist)
        axs[1].set_title("Weights")
            
        # axs[episode_idx,0].bar(step_idx, tdes)
        # axs[episode_idx,0].set_title("TDEs")

        # axs[episode_idx,1].bar(step_idx, tdes ** self._alpha)
        # axs[episode_idx,1].set_title("TDEs ** Alpha")

        # axs[episode_idx,2].bar(step_idx, weights)
        # axs[episode_idx,2].set_title("Weights")

        # axs[episode_idx,3].bar(step_idx, tdes*weights)
        # axs[episode_idx,3].set_title("TDEs x Weights")

        # axs[episode_idx,4].bar(step_idx, (tdes*weights) ** self._alpha)
        # axs[episode_idx,4].set_title("TDEs x Weights ** Alpha")

        print("Saving...")
        plt.savefig("./results/_latest_tde_plots/" + f"{self._current_episode}ep_plots.png")



import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import warnings
import logging
import matplotlib.pyplot as plt
from stable_baselines3.common.segment_tree import SumSegmentTree, MinSegmentTree

class CustomPrioritizedReplayBufferCumSum(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            alpha: float = .6,
            log_path: str = "",
            custom_buffer_size = None,
            use_importance_sampling: bool = True,
            disable_within_episode_weights: bool = False,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)
        
        # Overwrite buffer size with custom value if given
        if custom_buffer_size is not None:
            self.buffer_size = custom_buffer_size

        self._alpha = alpha
        self.use_importance_sampling = use_importance_sampling
        self.disable_within_episode_weights = disable_within_episode_weights

        # Setup sampling array
        self.sample_arange = np.arange(self.buffer_size * self.n_envs)

        # Setup episode storage
        self._current_episode = np.arange(1, n_envs+1)
        self.episodes = np.zeros((self.buffer_size, self.n_envs))
        
        # Setup td error, td sum and td cumsum storage
        self._max_td = 1
        self._last_cum_td = np.zeros(self.n_envs, )
        self._cum_td = np.zeros(n_envs,)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.td_errors = np.zeros((self.buffer_size, self.n_envs))

        # Setup timestep storage
        self._current_timestep = np.ones(n_envs,)
        self.timesteps = np.zeros((self.buffer_size, self.n_envs))

        # Setup tree structures
        it_capacity = 1
        while it_capacity < (self.buffer_size * self.n_envs):
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        # Setup logger
        self.log_path = log_path
        self.logger = logging.getLogger(__name__)
        # logging.basicConfig(filename=self.log_path + 'replay_value_prediction_model.log', encoding='utf-8', level=logging.DEBUG)
        self.logger.info("Weight prediction model summary\n---")

        print(f"{self._alpha=}")
        print(f"{self.disable_within_episode_weights=}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """

        assert len(set(self._current_episode)) == self.n_envs, "Multiple environments share an episode"
        idx = self.pos

        super().add(obs, next_obs, action, reward, done, infos)        
        
        self.td_errors[idx] = self._max_td
        self.episodes[idx] = self._current_episode
        self.timesteps[idx] = self._current_timestep

        cum_td = self._last_cum_td + self._max_td
        sum_td = self.cum_tds[idx]
        self.cum_tds[idx] = cum_td
        self.sum_tds[self.episodes==self._current_episode] = sum_td

        flat_idxes = self._flatten_idx(idx)
        self._it_sum[flat_idxes] = (cum_td/sum_td) ** self._alpha
        self._it_min[flat_idxes] = (cum_td/sum_td) ** self._alpha

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self._last_cum_td = self.cum_tds[idx] * ~done

    def _flatten_idx(self, idx):
        return idx + (np.arange(self.n_envs) * self.buffer_size)
    
    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self.observations) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx
    
    def sample(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()

        num_transitions_gathered = (self.buffer_size if self.full else self.pos) * self.n_envs

        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()

        weights = (p_sample * num_transitions_gathered) ** (-beta) / max_weight
        encoded_sample = super()._get_samples(idxes, env=env)

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), weights, idxes

    # def get_within_episode_weights(self):
    #     """
    #     calculates weights for each timestep quantifying the reliability of TD errors based on the amount and size of subsequent TD errors
    #     """
    #     weights = self.cum_tds / self.sum_tds
    #     weights = np.where(self.episodes==0, 0, weights)
    #     return weights
    
    # def sample(self, batch_size: int, beta: float = .5, env: Optional[VecNormalize] = None):
    #     """
    #     Sample a batch of experiences.

    #     compared to ReplayBuffer.sample
    #     it also returns importance weights and idxes
    #     of sampled experiences.

    #     :param batch_size: (int) How many transitions to sample.
    #     :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
    #     :param env: (Optional[VecNormalize]) associated gym VecEnv
    #         to normalize the observations/rewards when sampling
    #     :return:
    #         - obs_batch: (np.ndarray) batch of observations
    #         - act_batch: (numpy float) batch of actions executed given obs_batch
    #         - rew_batch: (numpy float) rewards received as results of executing act_batch
    #         - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
    #         - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
    #             and 0 otherwise.
    #         - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
    #             each sampled transition
    #         - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
    #     """
    #     assert beta > 0

    #     within_episode_weights = self.get_within_episode_weights()

    #     if self.disable_within_episode_weights:
    #         weighted_tdes = (self.td_errors) ** self._alpha
    #     else:
    #         weighted_tdes = (self.td_errors * within_episode_weights) ** self._alpha

    #     sampling_probs = weighted_tdes / weighted_tdes.sum()
    #     flat_sampling_probs = sampling_probs.reshape(-1)
    #     non_zero_mask = flat_sampling_probs!=0
    #     sample_idxes = np.random.choice(self.sample_arange[non_zero_mask], batch_size, p=flat_sampling_probs[non_zero_mask])
        
    #     row_idxes = sample_idxes//self.n_envs
    #     col_idxes = sample_idxes%self.n_envs
        
    #     sampling_probas_of_sample = flat_sampling_probs[sample_idxes]
    #     encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

    #     # Get importance sampling weights
    #     num_transitions_gathered = (self.buffer_size if self.full else self.pos) * self.n_envs
    #     p_min = sampling_probs[:num_transitions_gathered].min()
    #     max_weight = (p_min * num_transitions_gathered) ** (-beta)
    #     IS_weights = (sampling_probas_of_sample * num_transitions_gathered) ** (-beta) / max_weight

    #     return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)
    
    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        # assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes = idxes//self.n_envs
        col_idxes = idxes%self.n_envs

        # Check if a single transition was sampled multiple times in a single batch
        _, unique_idx = np.unique(np.vstack([row_idxes, col_idxes]), axis=1, return_index=True)
        if len(unique_idx) < len(priorities):
            priorities = priorities[unique_idx]
            row_idxes = row_idxes[unique_idx]
            col_idxes = col_idxes[unique_idx]
            warnings.warn("The same transition was sampled multiple times.")

        assert len(col_idxes) == len(priorities)
        assert len(row_idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.observations)
        assert (self.td_errors < 0).sum() == 0
        assert (self.cum_tds < 0).sum() == 0
        assert (self.sum_tds < 0).sum() == 0

        current_tds = self.td_errors[row_idxes, col_idxes]
        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = current_tds-priorities

        # Update TD Errors
        self.td_errors[row_idxes, col_idxes] = priorities

        # Update cumsum & sum
        change_idxes = np.array([[],[]], dtype=np.int64)
        for (episode_to_update, timestep_to_update, delta) in zip(episodes_to_update, timesteps_to_update, deltas):
            
            episode_mask = np.where(self.episodes == episode_to_update)
            episode_timestep_mask = np.where((self.episodes == episode_to_update) & (self.timesteps >= timestep_to_update))

            self.sum_tds[episode_mask] -= delta
            self.cum_tds[episode_timestep_mask] -= delta            
            change_idxes = np.hstack([change_idxes, np.array(episode_mask)])

        change_idxes = np.unique(change_idxes, axis=1)
        change_idxes_2d = change_idxes[0, :], change_idxes[1, :]
        change_idxes_flat = change_idxes[0, :] + change_idxes[1, :] * self.buffer_size

        # Overwrite tree storage
        new_priorities = (self.td_errors*self.cum_tds/self.sum_tds)[change_idxes_2d] ** self._alpha
        self._it_sum[change_idxes_flat] = new_priorities
        self._it_min[change_idxes_flat] = new_priorities

        self._max_td = max(self._max_td, np.max(priorities))

        
    def plot_sampling_distributions(self, tds_split_by_eps, weights_by_eps):

        assert self.n_envs==1, "Currently incompatible with multiple environments"

        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15, 15))
        all_tde_bin_means, all_weight_bin_means = [], []

        for episode_idx, (tdes, weights) in enumerate(zip(tds_split_by_eps, weights_by_eps)):
            
            if not tdes.sum() == 0:
                run_progress = np.arange(len(tdes)) / len(tdes)

                num_bins = 10
                bins = np.linspace(0, 1, num_bins)
                digitized = np.digitize(run_progress, bins)

                tde_bin_means = [tdes[digitized == i].mean() for i in range(1, len(bins))]
                weights_bin_means = [weights[digitized == i].mean() for i in range(1, len(bins))]

                all_weight_bin_means.append(weights_bin_means)
                all_tde_bin_means.append(tde_bin_means)
        
        weight_dist = np.nanmean(np.array(all_weight_bin_means), axis=0)
        tde_dist = np.nanmean(np.array(all_tde_bin_means), axis=0)
        run_progress = np.arange(0, len(weight_dist))

        axs[0].bar(run_progress, tde_dist)
        axs[0].set_title("TDEs")
        axs[1].bar(run_progress, weight_dist)
        axs[1].set_title("Weights")
            
        # axs[episode_idx,0].bar(step_idx, tdes)
        # axs[episode_idx,0].set_title("TDEs")

        # axs[episode_idx,1].bar(step_idx, tdes ** self._alpha)
        # axs[episode_idx,1].set_title("TDEs ** Alpha")

        # axs[episode_idx,2].bar(step_idx, weights)
        # axs[episode_idx,2].set_title("Weights")

        # axs[episode_idx,3].bar(step_idx, tdes*weights)
        # axs[episode_idx,3].set_title("TDEs x Weights")

        # axs[episode_idx,4].bar(step_idx, (tdes*weights) ** self._alpha)
        # axs[episode_idx,4].set_title("TDEs x Weights ** Alpha")

        print("Saving...")
        plt.savefig("./results/_latest_tde_plots/" + f"{self._current_episode}ep_plots.png")