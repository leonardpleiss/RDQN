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

class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            alpha: float = .6
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)

        assert n_envs == 1, "This implementation may currently not be compatible with multiple environments!"

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

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
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
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

        # print(f"{max_weight=}")
        # print(f"{p_min=}")
        # print(f"{np.quantile(p_sample, [.001, 0.25, .5, 0.75, .99])=}")
        # print(f"{np.quantile(weights, [.001, 0.25, .5, 0.75, .99])=}")
        # print("---")

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), weights, idxes
    
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
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.observations)
        
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))

class PrioritizedReplayBufferPropagating(ReplayBuffer):

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
            custom_buffer_size = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)

        # assert n_envs == 1, "This implementation may currently not be compatible with multiple environments!"

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        # Overwrite buffer size with custom value
        if custom_buffer_size is not None:
            self.buffer_size = custom_buffer_size

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
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
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
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.observations)

        prev_idxes = idxes-1
        eligibility_mask = prev_idxes>=0
        eli_prev_idxes = prev_idxes[eligibility_mask]

        sum_delta = self._it_sum[idxes] - priorities
        min_delta = self._it_min[idxes] - priorities
        
        self._it_sum[idxes] = (self._it_sum[idxes] - sum_delta) ** self._alpha
        self._it_min[idxes] = (self._it_min[idxes] - min_delta) ** self._alpha

        self._it_sum[eli_prev_idxes] = abs(self._it_sum[eli_prev_idxes] + sum_delta[eligibility_mask]) ** self._alpha
        self._it_min[eli_prev_idxes] = abs(self._it_min[eli_prev_idxes] + min_delta[eligibility_mask]) ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))

class CustomPrioritizedReplayBuffer(ReplayBuffer):

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
            max_episodic_decay: float = 1.,
            max_step_decay: float = 1.,
            est_episode_length: int = 400., 
            custom_buffer_size = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)

        # Overwrite buffer size with custom value
        if custom_buffer_size is not None:
            self.buffer_size = custom_buffer_size

        # assert n_envs == 1, "This implementation may currently not be compatible with multiple environments!"
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < self.buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        # Setup: Across-episode decay
        self.episodic_transition_age = np.zeros(self.buffer_size)

        # Setup: Within-episode decay
        self.last_episode_end_idx = -1
        self.steps_until_done = np.zeros(self.buffer_size)
        
        # Calculate decays
        est_episodes_in_buffer = self.buffer_size / est_episode_length
        
        self.episodic_decay = max_episodic_decay ** (1/est_episodes_in_buffer)
        self.steps_until_done_decay = max_step_decay ** (1/est_episode_length)

        self.episode_idxes = np.array([], dtype=np.int32)

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
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

        # Reset transition age for newly added transitions
        self.episodic_transition_age[idx] = 1

        self.episode_idxes = np.append(self.episode_idxes, np.int32(idx))
        
        if done:
            
            # Store number of steps until the end of the episode
            self.steps_until_done[self.episode_idxes] = np.arange(len(self.episode_idxes), 0, -1)

            # Store & update transition age for gathered transitions
            self.episodic_transition_age[self.episodic_transition_age > 0] += 1
            self.episodic_transition_age[self.episode_idxes] = 1

            # Apply within episode decay
            decayed_priorities = (
                self._it_min[self.episode_idxes] 
                * self.steps_until_done_decay ** self.steps_until_done[self.episode_idxes]
            )
            self._it_min[self.episode_idxes] = decayed_priorities
            self._it_sum[self.episode_idxes] = decayed_priorities

            # Apply across episode decay
            self._it_sum.multiply_all_nodes(self.episodic_decay)
            self._it_min.multiply_all_nodes(self.episodic_decay)

            # Clear last episode idxes
            self.episode_idxes = np.array([], dtype=np.int32)
    
    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, self.num_transitions_gathered - 1)
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

        self.num_transitions_gathered = (self.buffer_size if self.full else self.pos) * self.n_envs

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()

        max_weight = (p_min * self.num_transitions_gathered) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()

        weights = (p_sample * self.num_transitions_gathered) ** (-beta) / max_weight
        encoded_sample = super()._get_samples(idxes, env=env)

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), weights, idxes
    
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
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.observations)

        # Overwrite newly computed priorities
        updated_priorities = (priorities
                              ** self._alpha # alpha adjustment
                              * (self.episodic_decay ** self.episodic_transition_age[idxes]) # age adjustment
                              * (self.steps_until_done_decay ** self.steps_until_done[idxes]) # distance from done adjustment
                             )

        self._it_sum[idxes] = updated_priorities
        self._it_min[idxes] = updated_priorities

        self._max_priority = max(self._max_priority, np.max(priorities))

        # if self.full:
        #     print(self.episodic_transition_age.min(), self.episodic_transition_age.max())

class DynamicPrioritizedReplayBuffer(ReplayBuffer):

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
            predictor_model_type: str = "tree",
            train_set_size: int = 100_000,
            value_batch_proportion: float = .1,
            log_path: str = "",
            use_importance_sampling: bool = False,
            custom_buffer_size = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)
        
        # Overwrite buffer size with custom value
        if custom_buffer_size is not None:
            self.buffer_size = custom_buffer_size

        self.replay_value_predictor_model_type = predictor_model_type
        self.train_set_size = train_set_size
        self.value_batch_proportion = value_batch_proportion
        self.use_importance_sampling = use_importance_sampling

        self.metrics_map = {
            "num_steps": 0,
            "total_steps_in_run": 1,
            "td_errors": 2,
            "episodic_distance": 3,
            "immediate_reward": 4,
            "run_reward": 5,
            "run_completion_percentage": 6,
            "is_done": 7,
            "episodic_sample_distance": 8,
        } 
        self.n_features = len(self.metrics_map.keys())
        self.metrics_storage = np.zeros((self.n_features, self.buffer_size, self.n_envs))

        self.run_finished_mask = np.zeros((self.buffer_size, self.n_envs)).astype(bool)
        self.train_features = np.empty((0, self.n_features))
        self.train_labels = np.array([])

        self.last_performance_score = 0
        self.train_set_full = False

        self.max_replay_value = 1
        self.max_td = 1
        self.idx_arange = np.arange(self.buffer_size * self.n_envs)

        self.just_fitted_predictor = False

        self.log_path = log_path
        self.logger = logging.getLogger(__name__)


    def add_transition_metric(
            self,
            metric_name: str,
            metric: np.ndarray
    ) -> None:        
        """add transition metrics to the metrics storage"""

        idx = self.pos
        self.metrics_storage[self.metrics_map[metric_name]][idx] = np.array(metric)

    def add_run_metrics(
            self,
            dones,
    ) -> None:
        """add run metrics at the end of each run"""

        for col_idx, done in enumerate(dones):
            if done:
                idx = self.pos
                run_length = int(self.metrics_storage[self.metrics_map["num_steps"]][:idx, col_idx][-1]) + 1
                total_reward = self.metrics_storage[self.metrics_map["immediate_reward"]][:idx, col_idx][-run_length:].sum()
                self.metrics_storage[self.metrics_map["run_reward"]][:idx, col_idx][-run_length:] = total_reward
                self.metrics_storage[self.metrics_map["total_steps_in_run"]][:idx, col_idx][-run_length:] = run_length
                self.metrics_storage[self.metrics_map["run_completion_percentage"]][:idx, col_idx][-run_length:] = self.metrics_storage[self.metrics_map["num_steps"]][:idx, col_idx][-run_length:] / run_length
                self.run_finished_mask[:idx, col_idx][-run_length:] = True
            
    def _update_transition_metrics_after_sample(self, flat_idxes) -> None:
        """update transition metrics after sampling them"""
        batch_idxes, env_idxes = self._flat_idxes_to_2d(flat_idxes)
        self.metrics_storage[self.metrics_map["episodic_sample_distance"]][batch_idxes, env_idxes] = 0

    def insert_td_errors(self, td_errors, sample_idxes):
        sample_idxes_2d = self._flat_idxes_to_2d(sample_idxes)
        self.metrics_storage[self.metrics_map["td_errors"]][sample_idxes_2d] = td_errors.detach().numpy().flatten()
        self.max_td = self.metrics_storage[self.metrics_map["td_errors"]].max()

    def update_transition_metrics_after_run(
            self,
            dones,
    ) -> None:
        """update transition metrics at the end of each run"""

        idx = self.pos

        # update episodic distance
        self.metrics_storage[self.metrics_map["episodic_distance"]][:idx, dones] += 1
        self.metrics_storage[self.metrics_map["episodic_sample_distance"]][:idx, dones] += 1
        
    def sample(
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
            beta: float = 0.5,
        ) -> Tuple[ReplayBufferSamples, np.array, np.array]:

        # Obtain replay probabilities
        prio_replay_probabilities, even_replay_probabilities = self._get_replay_probabilities()

        # Prioritized & uniform sampling
        prio_batch_size = int(batch_size * self.value_batch_proportion)
        even_batch_size = batch_size - prio_batch_size
        sample_idxes = np.arange(self.buffer_size * self.n_envs) if self.full else np.arange(self.pos * self.n_envs)

        # Obtain sample indices
        flat_prio_idxes = np.random.choice(sample_idxes, size=prio_batch_size, p=prio_replay_probabilities)
        flat_even_idxes = np.random.choice(sample_idxes, size=even_batch_size)
        flat_idxes = np.concatenate((flat_prio_idxes, flat_even_idxes))
        
        # Compute weights for importance sampling
        if self.use_importance_sampling:
            probabilities_of_sample = np.concatenate((prio_replay_probabilities[flat_prio_idxes], even_replay_probabilities[flat_even_idxes]))
            min_replay_probability = prio_replay_probabilities.min() / prio_replay_probabilities.sum()
            num_transitions_gathered = (self.buffer_size if self.full else self.pos) * self.n_envs
            max_weight = (min_replay_probability * num_transitions_gathered) ** (-beta)
            weights = (probabilities_of_sample * num_transitions_gathered) ** (-beta) / max_weight
        else:
            weights = np.ones_like(flat_idxes)
        
        # Store and update features
        self._store_train_features(flat_idxes)
        self._update_transition_metrics_after_sample(flat_idxes)

        # Encode sample
        encoded_sample = self._get_samples(flat_idxes, env=env)

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), weights, flat_idxes
    
    def _flat_idxes_to_2d(self, flat_idxes: np.array) -> Tuple[np.array, np.array]:

        batch_idxes = flat_idxes//self.n_envs
        env_idxes = flat_idxes%self.n_envs

        return batch_idxes, env_idxes
    
    def _get_samples(self, flat_idxes: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        batch_idxes, env_idxes = self._flat_idxes_to_2d(flat_idxes)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_idxes + 1) % self.buffer_size, env_idxes, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_idxes, env_idxes, :], env)

        data = (
            self._normalize_obs(self.observations[batch_idxes, env_idxes, :], env),
            self.actions[batch_idxes, env_idxes, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_idxes, env_idxes] * (1 - self.timeouts[batch_idxes, env_idxes])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_idxes, env_idxes].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_replay_probabilities(
            self
        ) -> np.array:

        # Get masks
        if not self.full:
            timestep_played = np.tile(np.arange(self.buffer_size) < self.pos, (self.n_envs, 1)).T
        elif self.full:
            timestep_played = np.full((self.buffer_size * self.n_envs, 1), True)
        
        flat_timestep_played = timestep_played.reshape(-1)
    
        # Get replay values for played timesteps trials
        replay_values = self._predict_replay_value()[flat_timestep_played]

        # Overwrite replay values of transitions in unfinished runs to the max replay value
        self.max_replay_value = replay_values.max()
        trial_not_done = ~self.run_finished_mask[flat_timestep_played].reshape(-1)
        replay_values[trial_not_done] = self.max_replay_value

        # Get probabilities using softmax
        prio_replay_probabilities = softmax(replay_values)
        even_replay_probabilities = softmax(np.ones_like(prio_replay_probabilities))
        
        return prio_replay_probabilities, even_replay_probabilities

    def _store_train_features(self, flat_idxes):

        batch_idxes, env_idxes = self._flat_idxes_to_2d(flat_idxes)

        new_batch_features = self.metrics_storage[:, batch_idxes, env_idxes].T
        self.train_features = np.vstack([self.train_features, new_batch_features])

        if self.train_features.shape[0] >= self.train_set_size:
            self.train_features = self.train_features[-self.train_set_size:, :]
            self.train_set_full = True

    def _store_train_label(self, new_label):
        while self.train_features.shape[0] > len(self.train_labels):
            self.train_labels = np.append(self.train_labels, new_label)
        
        self.train_labels = self.train_labels[-self.train_set_size:]

        if self.train_set_full:
            self._update_replay_value_predictor()

    def _update_replay_value_predictor(self):
            
        if self.replay_value_predictor_model_type == "lsq":
            self.replay_value_predictor, residuals, _, _ = np.linalg.lstsq(self.train_features, self.train_labels, rcond=-1)

        elif self.replay_value_predictor_model_type == "tree":
            self.replay_value_predictor = DecisionTreeRegressor(max_depth=10)
            self.replay_value_predictor.fit(self.train_features, self.train_labels)
            residuals = None

        self._log_replay_value_predictor_results(residuals=residuals)

        # Empty training set
        self.train_features = np.empty((0, len(self.metrics_map.keys())))
        self.train_labels = np.array([])
        self.train_set_full = False

    def _predict_replay_value(self):

        if hasattr(self, 'replay_value_predictor'):

            if self.replay_value_predictor_model_type == "lsq":
                features = self.metrics_storage.transpose(1, 2, 0).reshape(self.buffer_size*self.n_envs, self.n_features)
                replay_values = np.matmul(features, self.replay_value_predictor)
        
            if self.replay_value_predictor_model_type == "tree":
                features = self.metrics_storage.transpose(1, 2, 0).reshape(self.buffer_size*self.n_envs, self.n_features)
                replay_values = self.replay_value_predictor.predict(features)

            if self.use_importance_sampling:
                # Avoid extremely small replay values
                replay_values = (replay_values - replay_values.min()) / (replay_values.max() - replay_values.min()) + 1e-6
                return replay_values
            
            # Log difference between high & low replay value transitions
            if self.just_fitted_predictor:

                num_transitions = max(1, len(replay_values) // 10)
                top_idxs = np.argsort(replay_values)[-num_transitions:]

                log_str = ""

                for feature_idx, feature_name in enumerate(self.metrics_map):

                    feature_column = features[:, feature_idx]
                    top_feature_column = feature_column[top_idxs]

                    top_mean, top_sd = top_feature_column.mean(), top_feature_column.std()
                    gen_mean, gen_sd = feature_column.mean(), feature_column.std()

                    log_str += f"{feature_name}: Top: {round(top_mean, 2)} ({round(top_sd, 2)}); Full: {round(gen_mean, 2)} ({round(gen_sd, 2)})\n"
                
                self.logger.info(log_str)
                self.logger.info("---")

                self.just_fitted_predictor = False

            return replay_values
        
        else:
            return np.ones(self.buffer_size * self.n_envs)
        
    def _log_replay_value_predictor_results(self, residuals=None):

        if self.replay_value_predictor_model_type == "lsq":
            r2 = 1-residuals / (self.train_labels.size * self.train_labels.var())
            weight_summary = {list( self.metrics_map.keys())[idx] : self.replay_value_predictor[idx] for idx in range(len(list( self.metrics_map.keys())))}
            self.logger.info(weight_summary)

        if self.replay_value_predictor_model_type == "tree":
            predicted_train_labels = self.replay_value_predictor.predict(self.train_features) 
            r2 = r2_score(self.train_labels, predicted_train_labels)

        print(f"Fitted new sampling model with r2-score of {round(r2,6)}")

        self.logger.info(f"r2_score: {r2}")
        self.just_fitted_predictor = True