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
        self.selection_mask = np.zeros((self.buffer_size, self.n_envs)).astype(bool)
        self.just_freed = np.zeros((self.buffer_size, self.n_envs)).astype(bool)

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

        can_be_selected = done or (reward != 0)

        self.selection_mask[self.pos] = can_be_selected
        self.just_freed[self.pos] = can_be_selected

        # Update
        self._current_timestep = np.where(done, 1, self._current_timestep + 1)

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Half the batch from masked indices (self.sampling_mask), other half uniformly at random
        over the populated buffer. Only the masked half unlocks predecessors.
        """

        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            raise ValueError("No data available to sample from.")

        half = batch_size // 2
        rest = batch_size - half  # handles odd batch sizes

        # ----- Part 1: masked sampling (from rows < upper_bound only)
        if half > 0:
            live_rows_mask = np.zeros_like(self.selection_mask, dtype=bool)
            live_rows_mask[:upper_bound, :] = True
            effective_mask = self.selection_mask & live_rows_mask

            valid_indices = np.argwhere(effective_mask)
            if len(valid_indices) < half:
                raise ValueError(
                    f"Not enough valid samples in sampling_mask ({len(valid_indices)}) for masked half={half}"
                )

            chosen_masked = valid_indices[np.random.choice(len(valid_indices), size=half, replace=False)]
            sel_rows, sel_cols = chosen_masked[:, 0], chosen_masked[:, 1]
        else:
            sel_rows = np.empty(0, dtype=int)
            sel_cols = np.empty(0, dtype=int)

        # ----- Part 2: uniform random over populated buffer (ignores mask)
        if rest > 0:
            rand_rows = np.random.randint(0, upper_bound, size=rest)
            rand_cols = np.random.randint(0, self.n_envs, size=rest)
        else:
            rand_rows = np.empty(0, dtype=int)
            rand_cols = np.empty(0, dtype=int)

        # ----- Combine
        row_idxes = np.concatenate([sel_rows, rand_rows])
        col_idxes = np.concatenate([sel_cols, rand_cols])

        # ----- Fetch samples
        encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

        # ----- Unlock predecessors ONLY for part-1 (masked) samples
        if sel_rows.size > 0:
            pre_rows = sel_rows - 1
            pre_rows = np.where(sel_rows == 0, self.buffer_size - 1, pre_rows)

            not_first = self.timesteps[sel_rows, sel_cols] != 1
            pre_rows = pre_rows[not_first]
            pre_cols = sel_cols[not_first]

            self.selection_mask[pre_rows, pre_cols] = True
            if hasattr(self, "just_freed"):
                self.just_freed[pre_rows, pre_cols] = True  # optional: keep priority flag

        return encoded_sample


    # def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    #     """
    #     Sample a batch of transitions only from positions where self.sampling_mask is True.
    #     """

    #     # Get valid indices where sampling is allowed
    #     valid_indices = np.argwhere(self.selection_mask)

    #     if len(valid_indices) < batch_size:
    #         raise ValueError(
    #             f"Not enough valid samples in sampling_mask ({len(valid_indices)}) for batch_size={batch_size}"
    #         )

    #     # Randomly pick batch_size valid indices (without replacement)
    #     chosen = valid_indices[np.random.choice(len(valid_indices), size=half_batch_size)]

    #     # Split into rows and cols
    #     row_idxes, col_idxes = chosen[:, 0], chosen[:, 1]

    #     # Fetch actual samples
    #     encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

    #     # Enable sampling of predecessors
    #     pre_row_idxes = row_idxes-1
    #     pre_row_idxes = np.where(row_idxes == 0, self.buffer_size-1, pre_row_idxes)
    #     not_first_timestep = self.timesteps[row_idxes, col_idxes] != 1
    #     pre_row_idxes = pre_row_idxes[not_first_timestep]
    #     pre_col_idxces = col_idxes[not_first_timestep]
    #     self.selection_mask[pre_row_idxes, pre_col_idxces] = True

    #     return encoded_sample
    

    # def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    #     """
    #     Sample a batch of transitions with priority for entries where self.just_freed is True.
    #     After being sampled with priority, just_freed entries are reset to False.
    #     """

    #     # Step 1: get indices of priority entries
    #     priority_indices = np.argwhere(self.just_freed)

    #     # If too many priorities, clip to batch_size
    #     if len(priority_indices) > batch_size:
    #         chosen_priority = priority_indices[np.random.choice(len(priority_indices), size=batch_size, replace=False)]
    #         chosen_normal = np.empty((0, 2), dtype=int)

    #         print(0)
    #     else:
    #         chosen_priority = priority_indices
    #         remaining = batch_size - len(chosen_priority)

    #         # Step 2: get normal valid indices (eligible for sampling, excluding already chosen priorities)
    #         valid_indices = np.argwhere(self.selection_mask)
    #         if remaining > len(valid_indices):
    #             raise ValueError(
    #                 f"Not enough valid samples: need {remaining}, but only {len(valid_indices)} available."
    #             )

    #         chosen_normal = valid_indices[np.random.choice(len(valid_indices), size=remaining, replace=False)]
    #         print(remaining)

       

    #     # Combine priority + normal samples
    #     chosen = np.vstack([chosen_priority, chosen_normal])

    #     # Split rows and cols
    #     row_idxes, col_idxes = chosen[:, 0], chosen[:, 1]

    #     # Step 3: reset just_freed entries that were sampled
    #     self.just_freed[row_idxes, col_idxes] = False

    #     # Step 4: fetch actual samples
    #     encoded_sample = super()._get_samples(row_idxes, env=env, env_indices=col_idxes)

    #     # Enable sampling of predecessors
    #     pre_row_idxes = row_idxes-1
    #     pre_row_idxes = np.where(row_idxes == 0, self.buffer_size-1, pre_row_idxes)
    #     not_first_timestep = self.timesteps[row_idxes, col_idxes] != 1
    #     pre_row_idxes = pre_row_idxes[not_first_timestep]
    #     pre_col_idxes = col_idxes[not_first_timestep]
    #     self.selection_mask[pre_row_idxes, pre_col_idxes] = True
    #     self.just_freed[pre_row_idxes, pre_col_idxes] = True

    #     return encoded_sample
