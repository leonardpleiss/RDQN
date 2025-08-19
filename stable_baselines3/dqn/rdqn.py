import warnings
from typing import Any, ClassVar, Dict, Type, TypeVar

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.dqn.dqn import DQN

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class RDQN(DQN):

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

        assert self.replay_buffer_class.__name__ in ["PositionalReplayBuffer"]

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for gradient_step in range(gradient_steps):

            replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
                
            with th.no_grad():
                
                next_actions_online = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                # DDQN Target
                next_q_values_ddqn = self.q_net_target(replay_data.next_observations).gather(1, next_actions_online)
                target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones.float()) * self.gamma * next_q_values_ddqn

                # Online Target
                next_q_values_online = self.q_net(replay_data.next_observations).gather(1, next_actions_online)
                target_q_values_online = replay_data.rewards + (1 - replay_data.dones.float()) * self.gamma * next_q_values_online

                # Staleness factor
                all_current_q_values_target = self.q_net_target(replay_data.observations)
                current_q_values_target = th.gather(all_current_q_values_target, dim=1, index=replay_data.actions.long())

                initial_td = abs(current_q_values_target - target_q_values_ddqn)
                current_td = abs(current_q_values - target_q_values_ddqn)

                learning_potential = current_td / initial_td
                learning_potential = current_td / (initial_td + 1e-8)
                learning_potential = th.nan_to_num(learning_potential, nan=0.0, posinf=1.0, neginf=0.0)
                learning_potential = th.clamp(learning_potential, min=0, max=1)

                frozen_target = target_q_values_ddqn

                adaptive_target = (
                    (1 - relative_episodic_position) * target_q_values_online + 
                    relative_episodic_position * target_q_values_ddqn
                )
                
                target_q_values_adj = (
                    learning_potential * frozen_target +
                    (1-learning_potential) * adaptive_target
                )

            loss = F.smooth_l1_loss(current_q_values, target_q_values_adj)

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
        tb_log_name: str = "RDQN",
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