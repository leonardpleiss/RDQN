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