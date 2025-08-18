import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

from copy import deepcopy
from collections import deque

th.autograd.set_detect_anomaly(True)

SelfDQN = TypeVar("SelfDQN", bound="RDQN")

def find_capped_softmax_boost_stable(row_vector, index, percentage):
    """
    Finds the value 'a' to add to a specific element of each row vector
    to increase its softmax probability by a given percentage, with a cap.
    The implementation includes numerical stability for large input values.

    Args:
        row_vector (th.Tensor): A tensor of shape (batch_size, num_features).
        index (th.Tensor): A tensor of shape (batch_size, 1) with the
                              index to modify for each row.
        percentage (th.Tensor): A tensor of shape (batch_size, 1) with the
                                   desired percentage increase for each row.

    Returns:
        th.Tensor: A tensor of shape (batch_size, 1) containing the value 'a'.
    """

    # 1. Apply numerical stability trick for softmax and apply softmax
    row_max = th.max(row_vector, dim=1, keepdim=True)[0]
    stable_row_vector = row_vector - row_max
    exp_stable_row_vector = th.exp(stable_row_vector)

    N = th.sum(exp_stable_row_vector, dim=1, keepdim=True)
    
    # Use th.gather to get the exponential of the element at the index
    exp_xi = th.gather(exp_stable_row_vector, dim=1, index=index)
    S_orig = exp_xi / N

    # Identify cases where S_orig is already effectively 1.0
    # In these cases, no further boost is possible, so 'a' should be 0.
    # Use a small threshold for numerical stability.
    # is_S_orig_one = (S_orig >= (1.0 - 1e-7)).squeeze(1) # Boolean tensor (batch_size,)
    will_exceed_1 = (S_orig >= 1 - percentage).squeeze(1)

    # 3. Calculate the capped increase
    requested_increase = percentage * S_orig
    max_increase_possible = 1.0 - S_orig
    actual_increase = th.min(requested_increase, max_increase_possible)
    
    # 4. Calculate the new target softmax probability
    S_new_target = S_orig + actual_increase
    
    # Ensure S_new_target does not become exactly 1 to avoid division by zero
    # Also ensure it's not less than a small positive value
    epsilon_clamp_lower = 1e-9
    epsilon_clamp_upper = 1e-9
    S_new_target = th.clamp(S_new_target, min=epsilon_clamp_lower, max=1.0-epsilon_clamp_upper)

    # 5. Use the closed-form solution
    # D = sum(e^x_k) for k != i (using stable vectors)
    D = N - exp_xi
    # E_i = e^x_i (using stable vectors)
    E_i = exp_xi

    # Add a small epsilon for numerical stability in division and logarithm.
    # This helps prevent log(0) and division by zero when D or (1 - S_new_target) are near zero.
    epsilon_numerical = 1e-10 # A very small positive number
    
    # Add epsilon to D to ensure numerator is not exactly zero when D is zero
    numerator_for_log = S_new_target * (D + epsilon_numerical)

    # Add epsilon to (1.0 - S_new_target) to ensure denominator is not exactly zero
    denominator_for_log = E_i * (1.0 - S_new_target + epsilon_numerical)

    # Ensure the denominator is strictly positive
    denominator_for_log = th.max(denominator_for_log, th.tensor(epsilon_numerical, device=row_vector.device))

    fraction_term = numerator_for_log / denominator_for_log

    # Ensure fraction_term is strictly positive before taking log
    # This is a final safeguard against any floating point issues leading to non-positive value
    fraction_term = th.max(fraction_term, th.tensor(epsilon_numerical, device=row_vector.device))
    a = th.log(fraction_term)

    # If S_orig was effectively 1, set 'a' to 0 as no boost is possible.
    # This overrides the potentially problematic log calculation for these edge cases.
    # Unsqueeze is_S_orig_one to match 'a's dimensions (batch_size, 1)
    # a = th.where(is_S_orig_one.unsqueeze(1), th.zeros_like(a), a)
    # a = th.where(will_exceed_1.unsqueeze(1), th.zeros_like(a), a)

    inf_vector = th.full_like(a, float('inf'))

    a = th.where(will_exceed_1.unsqueeze(1), inf_vector, a)

    return a


class RDQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        replay_buffer_log_path: str = "",
        to_scale_with_reliability = "loss",
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.stored_q_values = deque(maxlen=10_000)

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        assert to_scale_with_reliability in ["ddqn_blend_inv_epipos_dqn", "pure_online", "ddqn_blend_inv_epipos_wavg", "ddqn_blend_inv_epipos", "distance_scaled_step", "ddqn_blend_inv_v2", "ddqn_blend_inv_v6", "ddqn_blend_inv_v5", "ddqn_blend_inv_v3", "ddqn_blend_inv_v4", "importance_sampling4", "importance_sampling3", "importance_sampling2", "importance_sampling", "bootstrap_increase_scaling", "clip_max_discounted_return", "loss_scaling_floored", "range_based_clip", "max_softmax_ratio", "scaledown_with_threshold", "clipped_bootstrap","ddqn_blend_inv", "scaled_bootstrap2", "scaled_bootstrap", "ddqn_blend", "logsumexp", "hardsoft", "loss", "loss_2", "target", "clip", "clip2", "limit_target_mean_deviation", "weighted_avg", "clip_target_historically"]
        self.to_scale_with_reliability = to_scale_with_reliability

        self.max_bootstrap_increase = 0.

        self.max_subsequent_tds = 0.
        self.max_reward = 0.

        self.weight_history = []

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            if self.replay_buffer_class.__name__ in ["DR_UNI", "R_UNI"]:
                self.replay_buffer.update_reliabilities()

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        prop_progress = self.num_timesteps / self._total_timesteps
        
        losses = []
        for gradient_step in range(gradient_steps):

            if self.replay_buffer_class.__name__ in ["R_UNI"]:
                replay_data, reliability, sample_idxs, max_td, subsequent_tds, relative_episodic_position = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                reliability = th.from_numpy(reliability).to(self.device).unsqueeze(1).float()
                subsequent_tds = th.from_numpy(subsequent_tds).to(self.device).unsqueeze(1).float()
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

            elif self.replay_buffer_class.__name__ in ["DR_UNI"]:
                replay_data, reliability, sample_idxs, max_discounted_return, subsequent_tds = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, discount_factor=self.gamma)
                reliability = th.from_numpy(reliability).to(self.device).unsqueeze(1).float()
                subsequent_tds = th.from_numpy(subsequent_tds).to(self.device).unsqueeze(1).float()

            elif self.replay_buffer_class.__name__ in ["PositionalReplayBuffer"]:
                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()
                
            elif self.replay_buffer_class.__name__ == "ReplayBuffer":
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env) # type: ignore[union-attr]
                reliability = th.ones_like(replay_data.actions, requires_grad=False, device=self.device)# .unsqueeze(1).float()

            else:
                raise ValueError(f"Unknown buffer specified: {self.replay_buffer_class.__name__}")
            
            # Get current Q-values estimates
            all_current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(all_current_q_values, dim=1, index=replay_data.actions.long())

            with th.no_grad():
                # Compute the next Q-values using the target network
                all_next_q_values = self.q_net_target(replay_data.next_observations)

                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = all_next_q_values.max(dim=1)

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.to_scale_with_reliability == "range_based_clip":
                
                next_q = (1 - replay_data.dones) * self.gamma * next_q_values
                current_q = current_q_values
                reward = replay_data.rewards

                # Split error into reward-error and bootstrap-error
                td_error = next_q + reward - current_q
                reward_error = th.maximum(th.zeros_like(current_q), reward - current_q)
                bootstrap_error = td_error - reward_error

                # Clip bootstrapped error if positive
                max_bootstrap_error = ((th.max(all_current_q_values, dim=1)[0] - th.min(all_current_q_values, dim=1)[0]) * 2).unsqueeze(1)

                # print(f"{max_bootstrap_error.size()=}")
                upper_limit = max_bootstrap_error # * floored_reliability
                clipped_bootstrap_error = th.clamp(bootstrap_error, max=upper_limit)

                adj_bootstrap_error = th.minimum(clipped_bootstrap_error, bootstrap_error)
                adj_target_q_values = current_q + reward_error + adj_bootstrap_error

                
                mask = adj_bootstrap_error != bootstrap_error
                print(f"{mask.sum() / len(mask)=}")
                print(f"{max_bootstrap_error[mask]=}")
                print(f"{bootstrap_error[mask]=}")
                print(f"{clipped_bootstrap_error[mask]=}")
                print("---")

                loss = F.smooth_l1_loss(current_q_values, adj_target_q_values, reduction='none')

            if self.to_scale_with_reliability == "max_softmax_ratio":
                
                next_q = (1 - replay_data.dones) * self.gamma * next_q_values
                current_q = current_q_values
                reward = replay_data.rewards

                # Split error into reward-error and bootstrap-error
                td_error = next_q + reward - current_q
                reward_error = th.maximum(th.zeros_like(current_q), reward - current_q)
                bootstrap_error = td_error - reward_error

                # Clip bootstrapped error if positive
                target_increase = th.ones_like(current_q) * .9 + reliability * .1

                max_increase = find_capped_softmax_boost_stable(row_vector=all_current_q_values, index=replay_data.actions.long(), percentage=target_increase)

                clipped_bootstrap_error = th.clamp(bootstrap_error, max=max_increase)

                adj_target_q_values = current_q_values + reward_error + clipped_bootstrap_error
                
                if (self.num_timesteps%100==0) and (gradient_step==0):
                    mask = adj_target_q_values != target_q_values
                    if mask.sum() > 0:
                        print(f"{mask.sum() / len(mask)=}")
                        print(f"{all_current_q_values[mask.squeeze(1)]=}")
                        print(f"{th.softmax(all_current_q_values, dim=1)[mask.squeeze(1)]=}")
                        print(f"{max_increase[mask]=}")
                        print(f"{bootstrap_error[mask]=}")
                        print(f"{clipped_bootstrap_error[mask]=}")
                        print(f"{reward_error[mask]=}")
                        print(f"{target_q_values[mask]=}")
                        print(f"{adj_target_q_values[mask]=}")
                        print(f"{reliability[mask]=}")
                        print("---")
                
                
                loss = F.smooth_l1_loss(current_q_values, adj_target_q_values, reduction='none')

            if self.to_scale_with_reliability == "bootstrap_increase_scaling":

                next_q = (1 - replay_data.dones) * self.gamma * next_q_values
                current_q = current_q_values
                reward = replay_data.rewards

                # Split error into reward-error and bootstrap-error
                td_error = next_q + reward - current_q
                reward_error = th.maximum(th.zeros_like(current_q), reward - current_q)
                bootstrap_error = td_error - reward_error

                scaled_bootstrap_error = bootstrap_error * reliability ** .2
                final_bootstrap_error = th.minimum(bootstrap_error, scaled_bootstrap_error)

                adj_target_q_values = current_q + reward_error + final_bootstrap_error

                loss = F.smooth_l1_loss(current_q_values, adj_target_q_values, reduction='none')

                if (self.num_timesteps%1000==0) and (gradient_step==0):
                    print(reliability.min(), reliability.max())
                    loss_was_clipped = adj_target_q_values != target_q_values
                    print(f"Proportion clipped: {loss_was_clipped.sum() / len(loss_was_clipped)}")

                sums_correct = th.isclose(td_error, bootstrap_error + reward_error, atol=1e-4)
                if not sums_correct.all():
                    print(f"{td_error[sums_correct]=}")
                    print(f"{reward_error[sums_correct]=}")
                    print(f"{bootstrap_error[sums_correct]=}")
                    print("---")

                    import sys
                    sys.exit()

            if self.to_scale_with_reliability == "clipped_bootstrap":

                next_q = (1 - replay_data.dones) * self.gamma * next_q_values
                current_q = current_q_values
                reward = replay_data.rewards

                # Split error into reward-error and bootstrap-error
                td_error = next_q + reward - current_q
                reward_error = th.maximum(th.zeros_like(current_q), reward - current_q)
                bootstrap_error = td_error - reward_error

                # Clip bootstrapped error if positive
                max_bootstrap_error = th.max(bootstrap_error)
                floored_reliability = 0.5 + reliability * 0.5
                upper_limit = max_bootstrap_error * floored_reliability
                clipped_bootstrap_error = th.clamp(bootstrap_error, max=upper_limit)

                adj_bootstrap_error = th.minimum(clipped_bootstrap_error, bootstrap_error)
                adj_target_q_values = current_q + reward_error + adj_bootstrap_error

                if (self.num_timesteps%1000==0) and (gradient_step==0):
                    print(reliability.min(), reliability.max())
                    loss_was_clipped = adj_target_q_values != target_q_values
                    print(f"Proportion clipped: {loss_was_clipped.sum() / len(loss_was_clipped)}")

                sums_correct = th.isclose(td_error, bootstrap_error + reward_error, atol=1e-4)
                if not sums_correct.all():
                    print(f"{td_error[sums_correct]=}")
                    print(f"{reward_error[sums_correct]=}")
                    print(f"{bootstrap_error[sums_correct]=}")
                    print("---")

                    import sys
                    sys.exit()

                loss = F.smooth_l1_loss(current_q_values, adj_target_q_values, reduction='none')

            if self.to_scale_with_reliability == "importance_sampling":

                td_errors = abs(target_q_values - current_q_values)
                relative_sample_importance = (1 + td_errors) / (1  + subsequent_tds)
                weights = relative_sample_importance ** .3
                weights /= weights.mean()

                # print(f"{weights.min(), weights.mean(), weights.max()=}")

                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * weights

            if self.to_scale_with_reliability == "importance_sampling2": # works pretty well for Acro & CP, weak for LL

                relative_sample_importance = 1 / (1  + subsequent_tds)
                weights = relative_sample_importance ** .3
                weights /= weights.mean()

               #  print(f"{weights.min(), weights.mean(), weights.max()=}")

                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * weights

            if self.to_scale_with_reliability == "importance_sampling3": # works well for all - see T20
                # relative_sample_importance = 1 / (1  + subsequent_tds)

                self.max_subsequent_tds = max(self.max_subsequent_tds, th.max(subsequent_tds))

                weights = 1 / (1 + subsequent_tds / self.max_subsequent_tds)
                weights /= weights.mean()

                self.weight_history.append(weights)
                # if (self.num_timesteps%2048==0) and (gradient_step==0):

                #     import matplotlib.pyplot as plt

                #     plt.hist(self.weight_history, bins = 100)
                #     plt.title(f"After {self.num_timesteps} timesteps")
                #     plt.show()

                #     print(f"{weights.min(), weights.max()=}")
                #     self.weight_history = []

                # print(f"{weights.min(), weights.mean(), weights.max()=}")

                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * weights

            if self.to_scale_with_reliability == "importance_sampling4":

                self.max_subsequent_tds = max(self.max_subsequent_tds, th.max(subsequent_tds))
                weights = th.exp(-subsequent_tds / self.max_subsequent_tds)
                weights /= weights.mean()
                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * weights
                
            if self.to_scale_with_reliability == "scaledown_with_threshold":

                next_q = (1 - replay_data.dones) * self.gamma * next_q_values
                current_q = current_q_values
                reward = replay_data.rewards

                # Split error into reward-error and bootstrap-error
                td_error = next_q + reward - current_q
                reward_error = th.maximum(th.zeros_like(current_q), reward - current_q)
                bootstrap_error = td_error - reward_error

                # Scale error larger 1 back
                error_over_1 = th.maximum(th.zeros_like(bootstrap_error), bootstrap_error-1)
                error_under_1 = th.clamp(bootstrap_error, max=1)
                clipped_bootstrap_error = error_under_1 + error_over_1 * reliability
                assert th.allclose(bootstrap_error, error_over_1 + error_under_1)

                adj_bootstrap_error = th.minimum(clipped_bootstrap_error, bootstrap_error)
                adj_target_q_values = current_q + reward_error + adj_bootstrap_error

                if (self.num_timesteps%1000==0) and (gradient_step==0):
                    print(reliability.min(), reliability.max())
                    loss_was_clipped = adj_target_q_values != target_q_values
                    print(f"Proportion clipped: {loss_was_clipped.sum() / len(loss_was_clipped)}")

                sums_correct = th.isclose(td_error, bootstrap_error + reward_error, atol=1e-4)
                if not sums_correct.all():
                    print(f"{td_error[sums_correct]=}")
                    print(f"{reward_error[sums_correct]=}")
                    print(f"{bootstrap_error[sums_correct]=}")
                    print("---")

                    import sys
                    sys.exit()

                loss = F.smooth_l1_loss(current_q_values, adj_target_q_values, reduction='none')

            if self.to_scale_with_reliability == "scaled_bootstrap":

                next_ = (1 - replay_data.dones) * self.gamma * next_q_values
                current_ = current_q_values
                reward_ = replay_data.rewards

                # Bei decrease garnix machen
                # Bei increase:
                #   Die auf next zu attribuierende Erhöhung limitieren

                target = next_ + reward_

                
                keep_next = current_ - reward_
                inc_next = next_ - keep_next


                bootstrapped_qval_increase = next_q_values - (current_q_values - replay_data.rewards)
                bootstrapped_qval_maintenance = next_q_values - bootstrapped_qval_increase

                max_bootstrapped_qval_increase = max(bootstrapped_qval_increase)

                max_clip = max_bootstrapped_qval_increase * reliability
                bootstrapped_qval_increase_clipped = th.clamp(bootstrapped_qval_increase, max=max_clip)

                target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (bootstrapped_qval_maintenance + bootstrapped_qval_increase_clipped)
                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

                print(f"{bootstrapped_qval_increase=}")
                print(f"{bootstrapped_qval_maintenance=}")
                print(f"{next_q_values=}")
                assert (bootstrapped_qval_increase + bootstrapped_qval_maintenance == next_q_values).all()
                
                # mask = next_q_values_adj != next_q_values
                # # print(f"{subsequent_tds=}")
                # print(f"{max_next_clip[mask]=}")
                # print(f"{next_q_values_adj[mask]=}")
                # print(f"{next_q_values[mask]=}")
                # print(f"{target_q_values_adj[mask]=}")
                # print(f"{target_q_values[mask]=}")
                # print(f"{reliability[mask]=}")
                # print("---")

            if self.to_scale_with_reliability == "scaled_bootstrap_sub":

                self.stored_q_values.extend(subsequent_tds)
                max_next = max(self.stored_q_values)
                # print(len(self.stored_q_values))

                
                max_next_clip = max_next * reliability
                next_q_values_adj = th.clamp(next_q_values, max_next_clip)
                target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_adj
                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv": # works very well

                noise = subsequent_tds / subsequent_tds.max()

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    0.5 * target_q_values +
                    0.5 * noise * target_q_values + 
                    0.5 * (1-noise) * target_q_values_ddqn
                )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_v2": # works well, but not for LunarLander

                self.max_subsequent_tds = max(self.max_subsequent_tds, th.max(subsequent_tds))
                noise = subsequent_tds / self.max_subsequent_tds

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    noise * target_q_values + 
                    (1-noise) * target_q_values_ddqn
                )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_v3": #

                self.max_subsequent_tds = max(self.max_subsequent_tds, th.max(subsequent_tds))

                noise = subsequent_tds / self.max_subsequent_tds

                correction_factor = (1 - noise) ** 2

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    (1 - correction_factor) * target_q_values + 
                    correction_factor * target_q_values_ddqn
                )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_v4": #

                with th.no_grad():
                    # Get next values from online network
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    next_q_values_online, _ = all_next_q_values_online.max(dim=1)
                    next_q_values_online = next_q_values_online.reshape(-1, 1)

                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                noise = subsequent_tds / subsequent_tds.max()

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    0.5 * target_q_values_online +
                    0.5 * noise * target_q_values_online + 
                    0.5 * (1-noise) * target_q_values_ddqn
                )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_v5": #

                with th.no_grad():
                    # Get next values from online network
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    next_q_values_online, _ = all_next_q_values_online.max(dim=1)
                    next_q_values_online = next_q_values_online.reshape(-1, 1)

                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                noise = subsequent_tds / subsequent_tds.max()

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    0.5 * target_q_values_online + 0.5 * target_q_values_ddqn
                )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_v6": #

                with th.no_grad():
                    # Get next values from online network
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    next_q_values_online, _ = all_next_q_values_online.max(dim=1)
                    next_q_values_online = next_q_values_online.reshape(-1, 1)

                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                noise = subsequent_tds / subsequent_tds.max()

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    0.25 * target_q_values_online + 0.75 * target_q_values_ddqn
                ) # v7

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "pure_online":
                next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                loss = F.smooth_l1_loss(current_q_values, target_q_values_online, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_epipos_dqn": # best so far

                with th.no_grad():

                    next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # Adjusted Target
                    target_q_values_adj = (
                        0.5 * target_q_values + 
                        0.5 * (1-relative_episodic_position) * target_q_values + 
                        0.5 * relative_episodic_position * target_q_values_ddqn
                    )
                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_epipos": # best so far

                with th.no_grad():

                    next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # Adjusted Target
                    target_q_values_adj = (
                        0.5 * target_q_values_online + 
                        0.5 * (1-relative_episodic_position) * target_q_values_online + 
                        0.5 * relative_episodic_position * target_q_values_ddqn
                    )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend_inv_epipos_wavg":

                with th.no_grad():

                    relative_episodic_position = relative_episodic_position ** (1 / self.action_space.n)
                    
                    next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # Adjusted Target
                    target_q_values_adj = (
                        (1-relative_episodic_position) * target_q_values_online + 
                        relative_episodic_position * target_q_values_ddqn
                    )
                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "ddqn_blend":

                noise = subsequent_tds / subsequent_tds.max()

                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                target_q_values_adj = (
                    0.5 * target_q_values +
                    0.5 * noise * target_q_values_ddqn +
                    0.5 * (1-noise) * target_q_values
                )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "distance_scaled_step":

                noise = subsequent_tds / subsequent_tds.max()

                # Get online target
                with th.no_grad():
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    next_q_values_online, _ = all_next_q_values_online.max(dim=1)
                    next_q_values_online = next_q_values_online.reshape(-1, 1)
                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                # Get DDQN target
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                target_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                distances = 1 + abs(target_q_values_ddqn - target_q_values_online)

                print(distances)

                adjusted_change = (target_q_values_online - current_q_values) / distances

                target_q_values_adj = current_q_values + adjusted_change

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "logsumexp":

                progress_scaled_reliability = prop_progress + (1 - prop_progress) * reliability
                target_q_values_cons = th.logsumexp(all_next_q_values, dim=1, keepdim=True)
                target_q_values_adj = target_q_values * progress_scaled_reliability + target_q_values_cons * (1-progress_scaled_reliability)
                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

                # print(f"{target_q_values_cons=}")
                # print(f"{target_q_values=}")
                # print("------")
                # if prop_progress%.01==0:
                #     was_clipped = target_q_values_adj != target_q_values
                #     print(f"{prop_progress=}")
                #     print(f"{progress_scaled_reliability.min(), progress_scaled_reliability.mean(), progress_scaled_reliability.max()=}")
                    
                #     # print(f"{max_target=}")
                #     print(f"{target_q_values_adj[was_clipped]=}")
                #     print(f"{target_q_values[was_clipped]=}")
                #     print(f"{(was_clipped / batch_size).sum()=}")
                #     print("-----")

            if self.to_scale_with_reliability == "clip_target_historically":

                progress_scaled_reliability = prop_progress + (1 - prop_progress) * reliability
                max_target = current_q_values + max_td * progress_scaled_reliability
                target_q_values_adj = th.clamp(target_q_values, max=max_target)
                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

                # Logging
                was_clipped = target_q_values_adj != target_q_values

                if prop_progress%.01==0:
                    print(f"{prop_progress=}")
                    print(f"{progress_scaled_reliability.min(), progress_scaled_reliability.mean(), progress_scaled_reliability.max()=}")
                    
                    # print(f"{max_target=}")
                    print(f"{target_q_values_adj[was_clipped]=}")
                    print(f"{target_q_values[was_clipped]=}")
                    print(f"{(was_clipped / batch_size).sum()=}")
                    print("-----")

            if self.to_scale_with_reliability == "loss_scaling_floored":
                
                abs_loss = target_q_values - current_q_values

                base_loss = th.clamp(abs_loss, max=1)

                scaled_added_loss = (abs_loss - base_loss) * reliability

                adj_target_q_values = current_q_values + base_loss + scaled_added_loss

                # print(f"{abs_loss[:3]=}")
                # print(f"{base_loss[:3]=}")
                # print(f"{scaled_added_loss[:3]=}")

                loss = F.smooth_l1_loss(current_q_values, adj_target_q_values, reduction='none') * reliability
                # print("---------")

            if self.to_scale_with_reliability == "loss":
                if (self.num_timesteps%100==0) and (gradient_step==0):
                    print(f"{reliability.min(), reliability.mean(), reliability.max()=}")

                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * reliability

            if self.to_scale_with_reliability == "weighted_avg":

                ranks = (-subsequent_tds.flatten()).argsort().argsort() / (len(subsequent_tds)-1)

                # subsequent_tds = th.clamp(subsequent_tds, min=1e-6)
                # inv_subsequent_tds = 1 / subsequent_tds

                # weights = inv_subsequent_tds / th.sum(inv_subsequent_tds) 
                # weights = weights * len(weights)

                regularized_weights = .5 + ranks

                # print(f"{ranks=}")
                # print(f"{regularized_weights.min(), regularized_weights.mean(), regularized_weights.max()=}")
                
                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * regularized_weights.squeeze()
                loss = loss.sum()

            if self.to_scale_with_reliability == "hardsoft":
                
                soft_target_q_values = th.sum(th.softmax(all_next_q_values, dim=1) * all_next_q_values, dim=1, keepdim=True)

                # print(th.softmax(all_next_q_values, dim=1))
                # print(th.softmax(all_next_q_values, dim=1) * all_next_q_values)
                # print(f"{soft_target_q_values=}")
                # print(f"{target_q_values=}")
                target_q_values = target_q_values * reliability + soft_target_q_values * (1-reliability)
                # print(f"{target_q_values=}")
                # print("---")
                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')

            if self.to_scale_with_reliability == "loss_2":
                loss_raw = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
                loss_adj = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * reliability

                loss = .5 * loss_raw + .5 * loss_adj

            if self.to_scale_with_reliability == "target":
                loss = F.smooth_l1_loss(current_q_values, final_target_q_values, reduction='none')
            
            if self.to_scale_with_reliability == "clip":

                max_target = .5 * (current_q_values + max_td * reliability) + .5 * target_q_values
                clipped_target_q_values = th.clamp(target_q_values, max=max_target)

                loss = F.smooth_l1_loss(current_q_values, clipped_target_q_values, reduction='none')

                was_clipped = target_q_values!=clipped_target_q_values
                if (self.num_timesteps%1000==0) and (gradient_step == 0):
                    if was_clipped.sum() > 0:
                        print(f"{max_td=}")
                        print(f"{max_target[was_clipped]=}")
                        print(f"{reliability[was_clipped]=}")
                        print(f"{clipped_target_q_values[was_clipped]=}")
                        print(f"{target_q_values[was_clipped]=}")
                        print(f"{(target_q_values!=clipped_target_q_values).sum()/len(max_target)=}")
                        print("-------")
                    else:
                        print("no-clip")
                        print("-------")

            if self.to_scale_with_reliability == "clip2":    

                # change_lim = th.max(target_q_values - current_q_values) * 2
                change_lim = self.max_reward

                rel_clip = reliability * change_lim
                # target_q_values_clipped = th.clamp(target_q_values, max=current_q_values+rel_clip)
                target_q_values_clipped = th.clamp(target_q_values, max=current_q_values + rel_clip)

                final_target_q_values = th.minimum(target_q_values_clipped, target_q_values)
                loss = F.smooth_l1_loss(current_q_values, final_target_q_values, reduction='none')

                if (self.num_timesteps%1000==0) and (gradient_step==0):
                    print(reliability.min(), reliability.max())
                    loss_was_clipped = target_q_values != final_target_q_values
                    print(f"{change_lim=}")
                    print(f"{reliability[loss_was_clipped]=}")
                    print(f"{current_q_values[loss_was_clipped]=}")
                    print(f"{target_q_values[loss_was_clipped]=}")
                    print(f"{target_q_values_clipped[loss_was_clipped]=}")
                    print(f"{final_target_q_values[loss_was_clipped]=}")
                    print(f"{rel_clip[loss_was_clipped]=}")
                    print("------")

            if self.to_scale_with_reliability == "limit_target_mean_deviation":

                q_avg =  all_next_q_values.mean(dim=1).reshape(-1, 1)

                reliability = th.clamp(reliability, min=0.5)
                adj_next_q_values = (
                    0.5 * next_q_values + 
                    0.5 * ((next_q_values * reliability) + (q_avg * (1-reliability)))
                )
                reliability * next_q_values + (1-reliability) * all_next_q_values.mean(dim=1).reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * adj_next_q_values
                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * reliability

            if self.to_scale_with_reliability == "clip_max_discounted_return":

                max_clip = max_discounted_return * reliability

                clipped_target_q_values = th.clamp(target_q_values, max=max_clip) 

                loss = F.smooth_l1_loss(current_q_values, clipped_target_q_values, reduction='none')


                
                # print(f"{was_clipped.sum()/len(was_clipped)=}")
                # print(f"{max_clip[was_clipped]=}")
                # print(f"{reliability[was_clipped]=}")
                # print("---")

                if (self.num_timesteps%10==0) and (gradient_step == 0):
                    was_clipped = target_q_values!=clipped_target_q_values

                    if was_clipped.sum() > 0:
                        print(f"{was_clipped.sum()/len(was_clipped)=}")
                        print(f"{max_clip[was_clipped]=}")
                        print(f"{reliability[was_clipped]=}")
                        print(f"{clipped_target_q_values[was_clipped]=}")
                        print(f"{target_q_values[was_clipped]=}")
                        print("-------")
                    else:
                        print("no-clip")
                        print("-------")

            if self.to_scale_with_reliability == "keep_learning":

                with th.no_grad():
                    
                    next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # Staleness factor
                    all_current_q_values_target = self.q_net_target(replay_data.observations)
                    current_q_values_target = th.gather(all_current_q_values_target, dim=1, index=replay_data.actions.long())

                    initial_td = abs(current_q_values_target - target_q_values_ddqn)
                    current_td = abs(current_q_values - target_q_values_ddqn)

                    learning_potential = current_td / initial_td
                    learning_potential = th.clamp(learning_potential, min=0, max=1)

                    # print(f"{initial_td[0]=}")
                    # print(f"{current_td[0]=}")
                    # print(f"{staleness_factor[0]=}")
                    # print(f"{relative_episodic_position[0]=}")
                    # print(f"{scaling_factor[0]=}")
                    # print("========")

                    frozen_target = target_q_values_ddqn

                    adaptive_target = (
                        (1-relative_episodic_position) * target_q_values_online + 
                        relative_episodic_position * target_q_values_ddqn
                    )
                    
                    target_q_values_adj = (
                        learning_potential * frozen_target +
                        (1-learning_potential) * adaptive_target
                    )

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            if self.to_scale_with_reliability == "keep_learning2":

                with th.no_grad():
                    
                    next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # Staleness factor
                    all_current_q_values_target = self.q_net_target(replay_data.observations)
                    current_q_values_target = th.gather(all_current_q_values_target, dim=1, index=replay_data.actions.long())

                    initial_td = abs(current_q_values_target - target_q_values_ddqn)
                    current_td = abs(current_q_values - target_q_values_ddqn)

                    learning_potential = current_td / initial_td
                    learning_potential = th.clamp(learning_potential, min=0, max=1)
                    
                    # Get new target
                    frozen_weight = learning_potential + (1 - learning_potential) * relative_episodic_position

                    frozen_target = target_q_values_ddqn
                    adaptive_target = target_q_values_online

                    target_q_values_adj = (
                        frozen_weight * frozen_target +
                        (1 - frozen_weight) * adaptive_target
                    )

                    # print(f"{learning_potential[0]=}")
                    # print(f"{relative_episodic_position[0]=}")
                    # print(f"{adaptive_weight[0]=}")
                    # print("========")

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj, reduction='none')

            loss = th.mean(loss)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
                
            if self.replay_buffer_class.__name__ in ["ReaPER", "R_UNI"]:

                td_errors = current_q_values - target_q_values
                new_td_errors = np.abs(td_errors.detach().cpu().numpy().reshape(-1)) # + 1e-6
                # new_td_errors = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none').cpu().numpy().reshape(-1)

                self.replay_buffer.update_priorities(idxes=sample_idxs, new_td_errors=new_td_errors)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
    
    
    # def _store_transition(
    #     self,
    #     replay_buffer: ReplayBuffer,
    #     buffer_action: np.ndarray,
    #     new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
    #     reward: np.ndarray,
    #     dones: np.ndarray,
    #     infos: List[Dict[str, Any]],
    # ) -> None:
    #     """
    #     Store transition in the replay buffer.
    #     We store the normalized action and the unnormalized observation.
    #     It also handles terminal observations (because VecEnv resets automatically).

    #     :param replay_buffer: Replay buffer object where to store the transition.
    #     :param buffer_action: normalized action
    #     :param new_obs: next observation in the current episode
    #         or first observation of the episode (when dones is True)
    #     :param reward: reward for the current transition
    #     :param dones: Termination signal
    #     :param infos: List of additional information about the transition.
    #         It may contain the terminal observations and information about timeout.
    #     """
    #     # Store only the unnormalized version
    #     if self._vec_normalize_env is not None:
    #         new_obs_ = self._vec_normalize_env.get_original_obs()
    #         reward_ = self._vec_normalize_env.get_original_reward()
    #     else:
    #         # Avoid changing the original ones
    #         self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

    #     # Avoid modification by reference
    #     next_obs = deepcopy(new_obs_)
    #     # As the VecEnv resets automatically, new_obs is already the
    #     # first observation of the next episode
    #     for i, done in enumerate(dones):
    #         if done and infos[i].get("terminal_observation") is not None:
    #             if isinstance(next_obs, dict):
    #                 next_obs_ = infos[i]["terminal_observation"]
    #                 # VecNormalize normalizes the terminal observation
    #                 if self._vec_normalize_env is not None:
    #                     next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
    #                 # Replace next obs for the correct envs
    #                 for key in next_obs.keys():
    #                     next_obs[key][i] = next_obs_[key]
    #             else:
    #                 next_obs[i] = infos[i]["terminal_observation"]
    #                 # VecNormalize normalizes the terminal observation
    #                 if self._vec_normalize_env is not None:
    #                     next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

    #     # print(f"{self._last_original_obs=}")
    #     # print(f"{th.from_numpy(self._last_original_obs)=}")
    #     # print(f"{th.from_numpy(buffer_action)=}")
    #     current_q_values = self.q_net(th.from_numpy(self._last_original_obs))
    #     current_q_values = th.gather(current_q_values, dim=1, index=th.from_numpy(buffer_action).unsqueeze(1))

    #     # print(f"{th.from_numpy(next_obs)=}")

    #     with th.no_grad():
    #         # Compute the next Q-values using the target network
    #         all_next_q_values = self.q_net_target(th.from_numpy(next_obs))

    #         # Follow greedy policy: use the one with the highest value
    #         next_q_values, _ = all_next_q_values.max(dim=1)
    #         # Avoid potential broadcast issue
    #         next_q_values = next_q_values.reshape(-1, 1)

    #         # 1-step TD target
    #         target_q_values = th.from_numpy(reward_) + (1 - th.from_numpy(dones).float()) * self.gamma * next_q_values

    #     td_errors = current_q_values - target_q_values
    #     new_td_errors = np.abs(td_errors.detach().cpu().numpy().reshape(-1))

    #     replay_buffer.add(
    #         self._last_original_obs,  # type: ignore[arg-type]
    #         next_obs,  # type: ignore[arg-type]
    #         buffer_action,
    #         reward_,
    #         dones,
    #         infos,
    #         new_td_errors,
    #     )

    #     self.max_reward = max(self.max_reward, reward_)

    #     self._last_obs = new_obs
    #     # Save the unnormalized observation
    #     if self._vec_normalize_env is not None:
    #         self._last_original_obs = new_obs_

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        # Convert everything to tensors on the right device
        obs_tensor = th.from_numpy(self._last_original_obs).float().to(self.device)
        action_tensor = th.from_numpy(buffer_action).long().to(self.device)

        if self.replay_buffer_class.__name__ in ["DR_UNI", "R_UNI"]:

            current_q_values = self.q_net(obs_tensor)
            current_q_values = th.gather(current_q_values, dim=1, index=action_tensor.unsqueeze(1))

            with th.no_grad():
                next_obs_tensor = th.from_numpy(next_obs).float().to(self.device)
                all_next_q_values = self.q_net_target(next_obs_tensor)
                next_q_values, _ = all_next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)

                reward_tensor = th.from_numpy(reward_).float().to(self.device)
                dones_tensor = th.from_numpy(dones).float().to(self.device)
                target_q_values = reward_tensor + (1 - dones_tensor) * self.gamma * next_q_values

            td_errors = current_q_values - target_q_values
            new_td_errors = np.abs(td_errors.detach().cpu().numpy().reshape(-1))

            replay_buffer.add(
                self._last_original_obs,  # still numpy
                next_obs,                 # still numpy
                buffer_action,
                reward_,
                dones,
                infos,
                new_td_errors,
            )

            self.max_reward = max(self.max_reward, reward_)

        else:
            replay_buffer.add(
                self._last_original_obs,  # still numpy
                next_obs,                 # still numpy
                buffer_action,
                reward_,
                dones,
                infos,
            )

        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
