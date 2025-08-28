import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any
from gymnasium import spaces
import torch as th
import warnings

class ForceIncludeReplayBuffer(ReplayBuffer):

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            signal_ratio: float = .1,
    ):
        
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

        if n_envs != 1:
            warnings.warn("The use of this buffer on multiple environments was not tested and may not work properly!")

        assert optimize_memory_usage == False, "Memory optimization is not supported."

        self.prev_high_td_rows = np.array([]).astype(int)
        self.prev_high_td_cols = np.array([]).astype(int)

        self.high_td_rows = np.array([]).astype(int)
        self.high_td_cols = np.array([]).astype(int)

        self.force_include_prop_now = .05
        self.force_include_prop_prev = .05

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        """
        Sample a batch of transitions only from positions where self.sampling_mask is True.
        """

        av_samples = self.pos if not self.full else self.buffer_size

        force_include_slots_now = min(len(self.high_td_rows), int(self.force_include_prop_now * batch_size))
        force_include_slots_prev = min(len(self.prev_high_td_rows), int(self.force_include_prop_prev * batch_size))
        sample_slots = batch_size - force_include_slots_now - force_include_slots_prev

        # Random sampling of indices
        row_idxes = np.random.randint(av_samples, size=sample_slots)
        col_idxes = np.random.randint(self.n_envs, size=sample_slots)

        # print(force_include_slots_now)
        # print(force_include_slots_prev)
        # print(len(self.high_td_rows))
        # print(len(self.prev_high_td_rows))
        # print("--")

        high_td_row_idxes = np.array([])
        high_td_col_idxes = np.array([])
        prev_high_td_row_idxes = np.array([])
        prev_high_td_col_idxes = np.array([])

        if force_include_slots_now:
            # Get samples with currently high TDs
            high_td_row_idxes = self.high_td_rows[-force_include_slots_now:]
            high_td_col_idxes = self.high_td_cols[-force_include_slots_now:]

            # print(f"{high_td_row_idxes=}")
            # print(f"{self.high_td_rows[-force_include_slots_now:]}")

        if force_include_slots_prev:
            # Get samples of predecessors of previously high TD transitions
            prev_high_td_row_idxes = self.prev_high_td_rows[:force_include_slots_prev]
            prev_high_td_col_idxes = self.prev_high_td_cols[:force_include_slots_prev]

            # Delete used ones
            self.prev_high_td_rows = self.prev_high_td_rows[force_include_slots_prev:]
            self.prev_high_td_cols = self.prev_high_td_cols[force_include_slots_prev:]

        # Merge with sampled indices
        row_idxes = np.hstack([row_idxes, high_td_row_idxes, prev_high_td_row_idxes]).astype(int)
        col_idxes = np.hstack([col_idxes, high_td_col_idxes, prev_high_td_col_idxes]).astype(int)

        # Fetch actual samples
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        sample_idxes = row_idxes, col_idxes

        return encoded_sample, sample_idxes
    
    def add_force_includes_by_index(self, row, col):

        self.high_td_rows = np.hstack([self.high_td_rows, row])
        self.high_td_cols = np.hstack([self.high_td_cols, col])

    def update_on_target_update(self):

        self.prev_high_td_rows = self.high_td_rows.copy() - 1 # Not accounting for episode overhang
        self.prev_high_td_cols = self.high_td_cols.copy()

        self.high_td_rows = np.array([])
        self.high_td_cols = np.array([])
