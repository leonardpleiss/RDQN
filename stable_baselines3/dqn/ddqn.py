import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.dqn.dqn import DQN

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DDQN(DQN):

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

        assert self.replay_buffer_class.__name__ in ["ReplayBuffer", "SelectiveReplayBuffer"]

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for gradient_step in range(gradient_steps):
                
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                
            with th.no_grad():
                # Get the action from the Q-network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                # Get the corresponding Q-value from the target network
                all_next_q_values = self.q_net_target(replay_data.next_observations)

                # Avoid potential broadcast issue
                next_q_values = all_next_q_values.gather(dim=1, index=next_actions).reshape(-1, 1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # print(f"{self.q_net(replay_data.next_observations)[:3]=}")
                # print(f"{next_actions[:3]=}")
                # print(f"{all_next_q_values[:3]=}")
                # print(f"{next_q_values[:3]=}")
                # print("----")

                # print(self.replay_buffer.buffer_size)
            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

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