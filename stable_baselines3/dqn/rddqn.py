import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers_custom import PrioritizedReplayBuffer, CustomPrioritizedReplayBuffer, PrioritizedReplayBufferPropagating
from stable_baselines3.common.buffers_custom_v2 import CustomPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSumProp, CustomPrioritizedReplayBufferCumSum2, CustomPrioritizedReplayBufferCumSum3, CustomPrioritizedReplayBufferCumSum4, CustomPrioritizedReplayBufferCumSum5, CustomPropagatingPrioritizedReplayBuffer, CustomPropagatingPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSum6, CustomPrioritizedReplayBufferCumSum7
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.dqn.dqn import DQN

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class RDDQN(DQN):
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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for _ in range(gradient_steps):

            if self.replay_buffer_class.__name__ in ["R_UNI", "ReaPER"]:
                start_beta = .4
                end_beta = 1.
                beta_increment = end_beta - start_beta
                beta = start_beta + (self.num_timesteps / self._total_timesteps) * beta_increment
                replay_data, reliability, sample_idxs = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, beta=beta)
                reliability = th.from_numpy(reliability).to(self.device).unsqueeze(1).float()
                
            elif self.replay_buffer_class.__name__ == "ReplayBuffer":
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env) # type: ignore[union-attr]
                reliability = th.ones_like(replay_data.actions, requires_grad=False, device=self.device)# .unsqueeze(1).float()

            else:
                raise ValueError(f"Unknown buffer specified: {self.replay_buffer_class.__name__}")
                
            with th.no_grad():
                # Get the action from the Q-network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                # Get the corresponding Q-value from the target network
                next_q_values = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            # loss_raw = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            # loss_adj = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * reliability
            # loss = th.mean(th.min(loss_raw, loss_adj))
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none') * reliability 

            # print(f"{loss_raw=}")
            # print(f"{loss_adj=}")
            # print(f"{th.min(loss_raw, loss_adj)=}")

            # loss = th.mean(F.smooth_l1_loss(current_q_values, target_q_values, reduce=None) * reliability)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.replay_buffer_class.__name__ in ["ReaPER", "R_UNI"]:

                reward_ratios = (abs(replay_data.rewards) / (1e-8 + abs(replay_data.rewards) + (1 - replay_data.dones) * self.gamma * abs(next_q_values))).detach().cpu().numpy().reshape(-1)
                td_errors = current_q_values - target_q_values
                new_td_errors = np.abs(td_errors.detach().cpu().numpy().reshape(-1)) + 1e-6
                self.replay_buffer.update_priorities(idxes=sample_idxs, new_td_errors=new_td_errors, reward_ratios=reward_ratios)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDQN",
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