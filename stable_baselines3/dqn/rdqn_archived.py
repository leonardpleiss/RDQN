import warnings
from typing import Any, ClassVar, Dict, Type, TypeVar, Union, Optional, Tuple

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.dqn.dqn import DQN

from torch import nn

SelfDQN = TypeVar("SelfDQN", bound="DQN")

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = pred - target
        abs_error = th.abs(error)

        quadratic = th.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()

def sigmoid_scale(vec, alpha=.2):
    vec = th.as_tensor(vec, dtype=th.float32)
    out = th.ones_like(vec)
    mask = vec < 0
    out[mask] = 2 * th.sigmoid(alpha * vec[mask])  # in (0,1), hits 1 at x=0
    return out

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
        target = "blend"
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
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
            replay_buffer_log_path,
        )

        self.target = target
        self.train_counter = 0

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        assert self.replay_buffer_class.__name__ in ["PositionalReplayBuffer"]

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for _ in range(gradient_steps):

            replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())


            if self.target == "alignment_discards":

                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size * 2, env=self._vec_normalize_env)

                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)

                    online_err_increase_raw = th.abs(new_online_err) - th.abs(online_err)

                    _, batch_idxes = th.topk(online_err_increase_raw.squeeze(), k=batch_size, largest=False)

                loss = F.smooth_l1_loss(current_q_values[batch_idxes], target_q_values_ddqn[batch_idxes])

            if self.target == "archive_20250826_DDQN_BASELINE_PositionalReplayBuffer_32bs_0seed_100evalfreq_1_alignment_IS/RDQN_28": # best?
                
                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_change = target_q_values_ddqn - current_q_values
                    online_change = target_q_values_online - current_q_values

                    dev = th.abs(ddqn_change - online_change)
                    dev_denom = th.abs(target_q_values_ddqn) + th.abs(target_q_values_online) + 1e-6

                    dev = dev / dev_denom
                    
                    min_, max_ = dev.min(), dev.max()
                    if min_ == max_:
                        relative_importance = th.ones(batch_size).reshape(-1, 1)
                    else:
                        ranks = th.argsort(th.argsort(dev.squeeze(), descending=True)) / len(dev)
                        relative_importance = 0.8 + (ranks - ranks.min()) / (ranks.max() - ranks.min()) * 0.4
                        

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * relative_importance
                loss = loss.mean()

            if self.target == "alignment_IS_v2":

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_change = target_q_values_ddqn - current_q_values
                    online_change = target_q_values_online - current_q_values

                    dev = th.abs(ddqn_change - online_change)
                    # dev_denom = th.abs(target_q_values_ddqn) + th.abs(target_q_values_online) + 1 # current_q_values
                    # dev = dev / dev_denom

                    min_, max_ = dev.min(), dev.max()
                    if min_ == max_:
                        relative_importance = th.ones(batch_size).reshape(-1, 1)
                    else:
                        ranks = th.argsort(th.argsort(dev.squeeze(), descending=True)) / len(dev)
                        relative_importance = 0.8 + (ranks - ranks.min()) / (ranks.max() - ranks.min()) * .4

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * relative_importance
                loss = loss.mean()

            if self.target == "alignment_IS_v4":

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_change = target_q_values_ddqn - current_q_values
                    online_change = target_q_values_online - current_q_values

                    # dev = th.abs(ddqn_change - online_change)
                    dev = th.sum(th.abs(all_next_q_values_online - all_next_q_values_offline), dim=1).reshape(-1, 1)
                    # denom = th.sum(th.abs(all_next_q_values_offline), dim=1).reshape(-1, 1)
                    # dev = dev / dev_denom

                    min_, max_ = dev.min(), dev.max()
                    if min_ == max_:
                        relative_importance = th.ones(batch_size).reshape(-1, 1)
                    else:
                        ranks = th.argsort(th.argsort(dev.squeeze(), descending=True)) / len(dev)
                        relative_importance = 0.8 + (ranks - ranks.min()) / (ranks.max() - ranks.min()) * .4

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * relative_importance
                loss = loss.mean()

            if self.target == "alignment_IS_v3":

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_change = target_q_values_ddqn - current_q_values
                    online_change = target_q_values_online - current_q_values

                    dev = th.abs(ddqn_change - online_change)
                    
                    min_, max_ = dev.min(), dev.max()
                    if min_ == max_:
                        relative_importance = th.ones(batch_size).reshape(-1, 1)
                    else:
                        ranks = th.argsort(th.argsort(dev.squeeze(), descending=True)) / len(dev)
                        relative_importance = 0.8 + (ranks - ranks.min()) / (ranks.max() - ranks.min()) * .4

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * relative_importance
                loss = loss.mean()

            if self.target == "alignment_IS_v7":

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)

                    # online_err_increase = th.abs(online_err) - th.abs(new_online_err)
                    online_err_increase_raw = th.abs(new_online_err) - th.abs(online_err)
                    online_err_increase = th.clamp(online_err_increase_raw, min=0, max=1)
                    downweight = online_err_increase / 10 # Scale in 0. - .1

                    weights = th.ones_like(downweight) - downweight
                    weights /= weights.mean()

                    perc_cut = (weights != 1).sum() / batch_size
                    max_raw_ol_inc = online_err_increase_raw.max()
                    weight_min = weights.min()
                    weight_max = weights.max()



                    self.logger.record("custom/perc_cut", perc_cut.item())
                    self.logger.record("custom/max_raw_ol_inc", max_raw_ol_inc.item())
                    self.logger.record("custom/weight_min", weight_min.item())
                    self.logger.record("custom/weight_max", weight_max.item())

                    # online_err_increase
                    # print(ddqn_err)
                    # print(online_err)
                    # print(new_online_err)

                    # print(online_err_increase)
                    # print("---")
                    

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * weights
                loss = loss.mean()

            if self.target == "alignment_IS_v8": # PROMISING!

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)

                    # online_err_increase = th.abs(online_err) - th.abs(new_online_err)
                    online_err_increase_raw = th.abs(new_online_err) - th.abs(online_err)
                    online_err_increase = th.clamp(online_err_increase_raw, min=0, max=1)
                    downweight = online_err_increase / 2 # Scale in 0. - .1

                    weights = th.ones_like(downweight) - downweight
                    weights /= weights.mean()

                    perc_cut = (weights != 1).sum() / batch_size
                    max_raw_ol_inc = online_err_increase_raw.max()
                    weight_min = weights.min()
                    weight_max = weights.max()

                    self.logger.record("custom/perc_cut", perc_cut.item())
                    self.logger.record("custom/max_raw_ol_inc", max_raw_ol_inc.item())
                    self.logger.record("custom/weight_min", weight_min.item())
                    self.logger.record("custom/weight_max", weight_max.item())

                    # online_err_increase
                    # print(ddqn_err)
                    # print(online_err)
                    # print(new_online_err)

                    # print(online_err_increase)
                    # print("---")
                    

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * weights
                loss = loss.mean()

            if self.target == "alignment_IS_v5":

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)

                    # online_err_increase = th.abs(online_err) - th.abs(new_online_err)
                    online_err_increase_raw = th.abs(new_online_err) - th.abs(online_err)
                    online_err_increase = th.clamp(online_err_increase_raw, min=0, max=1)
                    downweight = online_err_increase / 5 # Scale in 0. - .1

                    weights = th.ones_like(downweight) - downweight
                    weights /= weights.mean()

                    perc_cut = (weights != 1).sum() / batch_size
                    max_raw_ol_inc = online_err_increase_raw.max()
                    weight_min = weights.min()
                    weight_max = weights.max()



                    self.logger.record("custom/perc_cut", perc_cut.item())
                    self.logger.record("custom/max_raw_ol_inc", max_raw_ol_inc.item())
                    self.logger.record("custom/weight_min", weight_min.item())
                    self.logger.record("custom/weight_max", weight_max.item())

                    # online_err_increase
                    # print(ddqn_err)
                    # print(online_err)
                    # print(new_online_err)

                    # print(online_err_increase)
                    # print("---")
                    

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * weights
                loss = loss.mean()

            if self.target == "max_step_size":

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_change = target_q_values_ddqn - current_q_values
                    # print(ddqn_change)
                    online_change = th.abs(target_q_values_online - current_q_values)

                    on_off_dev = th.abs(online_change - ddqn_change)

                    target_q_values_adj = current_q_values + (ddqn_change / (1 + on_off_dev))

                loss = F.smooth_l1_loss(current_q_values, target_q_values_adj)

            if self.target == "alignment_IS":
                
                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # dev = th.abs(target_q_values_ddqn - target_q_values_online) / ((target_q_values_ddqn + target_q_values_online) / 2)

                    # dev = th.sum(th.abs(all_next_q_values_online - all_next_q_values_offline), dim=1).reshape(-1, 1)
                    # dev = dev / (all_next_q_values_online + all_next_q_values_offline) / 2

                    # dev = th.sum(th.abs(all_next_q_values_online - all_next_q_values_offline), dim=1).reshape(-1, 1)
                    # dev_denom = th.sum(th.abs(all_next_q_values_online + all_next_q_values_offline), dim=1).reshape(-1, 1) / 2
                    # dev_denom = th.sum((th.abs(all_next_q_values_online) + th.abs(all_next_q_values_offline)) / 2, dim=1).reshape(-1, 1)

                    # dev = th.sum(th.abs(all_next_q_values_online - all_next_q_values_offline), dim=1).reshape(-1, 1)
                    # dev_denom = th.sum(th.abs(all_next_q_values_online) + th.abs(all_next_q_values_offline), dim=1).reshape(-1, 1) / 2

                    ddqn_change = target_q_values_ddqn - current_q_values
                    online_change = target_q_values_online - current_q_values

                    dev = th.abs(ddqn_change - online_change)
                    dev_denom = th.abs(target_q_values_ddqn) + th.abs(target_q_values_online) + 1e-8 # current_q_values
                    # dev = dev / dev_denom

                    dev_normalized = (dev / dev_denom)

                    # Compute percentiles
                    lo = th.quantile(dev_normalized, .01)
                    hi = th.quantile(dev_normalized, .99)
                    dev_normalized = th.clamp(dev_normalized, lo, hi)

                    # relative_importance = relative_importance / relative_importance.mean()
                    
                    min_, max_ = dev.min(), dev.max()
                    if min_ == max_:
                        relative_importance = th.ones(batch_size).reshape(-1, 1)
                    # else:
                    #     ranks = th.argsort(th.argsort(dev.squeeze(), descending=True)) / len(dev)
                    #     relative_importance = 0.8 + (ranks - ranks.min()) / (ranks.max() - ranks.min()) * .4
                    else:
                        dev_normalized = th.log(1 + dev_normalized)
                        relative_importance = 0.8 + (dev_normalized - dev_normalized.min()) / (dev_normalized.max() - dev_normalized.min()) * .4
                        relative_importance = relative_importance / relative_importance.mean()
                   

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * relative_importance
                loss = loss.mean()

                self.logger.record("custom/imp_max", relative_importance.max().item())
                self.logger.record("custom/imp_min", relative_importance.min().item())
                self.logger.record("custom/imp_mean", relative_importance.mean().item())
                self.logger.record("custom/imp_median", relative_importance.median().item())
                self.logger.record("custom/imp_p25", th.quantile(relative_importance, 0.25).item())
                self.logger.record("custom/imp_p75", th.quantile(relative_importance, 0.75).item())

            if self.target == "capped_reduction":
                
                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    ddqn_change = target_q_values_ddqn - current_q_values
                    online_change = target_q_values_online - current_q_values

                    # ddqn_too_low = (ddqn_change < 0) * (online_change > 0)
                    is_aligned = (ddqn_change * online_change) > 0

                    self.logger.record("custom/ddqn_too_low_prop", (is_aligned.sum().item() / batch_size ))

                    smoothened_target = (
                        0.5 * target_q_values_ddqn + 0.5 * target_q_values_online
                    )

                    # target_q_values_adj = th.where(ddqn_too_low, current_q_values + (ddqn_change * 0.99), current_q_values + ddqn_change)
                    
                    target_q_values_adj = th.where(is_aligned, target_q_values_ddqn, smoothened_target)

            if self.target == "max_stepsize":
                
                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    # max_stepsize = th.abs(
                    #     th.stack([
                    #         target_q_values_online - current_q_values,
                    #         target_q_values_dqn - current_q_values
                    #     ]) 
                    # ).max() * 2.5

                    max_stepsize = th.abs(target_q_values_online - current_q_values) + th.abs(target_q_values_dqn - current_q_values)
                    # max_stepsize = th.abs(target_q_values_dqn - current_q_values).max()

                    ddqn_change = target_q_values_ddqn - current_q_values
                    ddqn_change_clipped = th.clamp(ddqn_change, -max_stepsize, max_stepsize)

                    prop_clipped = th.sum(ddqn_change != ddqn_change_clipped) / batch_size
                    sum_clipped = th.sum(ddqn_change - ddqn_change_clipped)
                    max_clipped = th.min(ddqn_change - ddqn_change_clipped)

                    self.logger.record("custom/prop_clipped", prop_clipped.item())
                    self.logger.record("custom/sum_clipped", sum_clipped.item())
                    self.logger.record("custom/max_clipped", max_clipped.item())
                    # self.logger.record("custom/max_stepsize", max_stepsize.item())

                    # if prop_clipped > 0:
                    #     print(f"{sum_clipped=}")
                    #     print(f"{max_clipped=}")
                    #     print(f"{max_stepsize=}")
                    #     print("=================")

                    target_q_values_adj = current_q_values + ddqn_change_clipped

            if self.target == "drift":

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Usefulness of DDQN target
                    ddqn_learning_potential = th.abs(current_q_values - target_q_values_ddqn)
                    target_drift = th.mean(th.abs(all_next_q_values_online - all_next_q_values_offline), dim=1).reshape(-1, 1)

                    online_factor = target_drift / (ddqn_learning_potential + target_drift + 1e-8)

                    # online_factor = th.clamp(online_factor, max=.5)
                    online_factor = th.clamp(online_factor-.5, min = 0)

                    self.logger.record("custom/target_drift", th.mean(target_drift).item())
                    # self.logger.record("custom/ddqn_online_deviation_max", ddqn_ol_dev_max)
                    # self.logger.record("custom/ddqn_online_deviation_min", ddqn_ol_dev_min)

                    self.logger.record("custom/ddqn_learning_potential", th.mean(ddqn_learning_potential).item())
                    self.logger.record("custom/online_factor", th.mean(online_factor).item())
                    self.logger.record("custom/online_factor_min", th.min(online_factor).item())
                    self.logger.record("custom/online_factor_max", th.max(online_factor).item())

                    target_q_values_adj = (
                        online_factor * target_q_values_online + 
                        (1 - online_factor) * target_q_values_ddqn
                    )

            if self.target == "tst":
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    # online_increase = (target_q_values_online - current_q_values) > 0
                    # ddqn_increase = (target_q_values_ddqn - current_q_values) > 0
                    # is_aligned = ((target_q_values_ddqn - current_q_values) * (target_q_values_online - current_q_values)) > 0

                    # ddqn_ol_dev_mean = th.mean(target_q_values_ddqn - target_q_values_online).item()
                    # ddqn_ol_dev_max = th.max(target_q_values_ddqn - target_q_values_online).item()
                    # ddqn_ol_dev_min = th.min(target_q_values_ddqn - target_q_values_online).item()

                    ddqn_dqn_dev_mean = th.mean(target_q_values_ddqn - target_q_values_dqn).item()
                    ddqn_dqn_dev_max = th.max(target_q_values_ddqn - target_q_values_dqn).item()

                    ddqn_dqn_dev_min_idx = th.argmin(target_q_values_ddqn - target_q_values_dqn)

                    diffs = target_q_values_ddqn - target_q_values_dqn                    
                    # self.logger.record("train/online_increase", th.sum(online_increase).item() / batch_size)
                    # self.logger.record("train/ddqn_increase", th.sum(ddqn_increase).item() / batch_size)
                    # self.logger.record("train/is_aligned", th.sum(is_aligned).item() / batch_size)
                    # self.logger.record("train/ddqn_dqn_decrease", ddqn_dqn_dev)
                    # self.logger.record("custom/ddqn_online_deviation_mean", ddqn_ol_dev_mean)
                    # self.logger.record("custom/ddqn_online_deviation_max", ddqn_ol_dev_max)
                    # self.logger.record("custom/ddqn_online_deviation_min", ddqn_ol_dev_min)

                    self.logger.record("custom/ddqn_dqn_deviation_mean", ddqn_dqn_dev_mean)
                    self.logger.record("custom/ddqn_dqn_deviation_max", ddqn_dqn_dev_max)
                    self.logger.record("custom/ddqn_dqn_deviation_min", diffs[ddqn_dqn_dev_min_idx].item())

                    self.logger.record("custom/ddqn_min_val", target_q_values_ddqn[ddqn_dqn_dev_min_idx].item())
                    self.logger.record("custom/dqn_min_val", target_q_values_dqn[ddqn_dqn_dev_min_idx].item())
                    self.logger.record("custom/curr_min_val", current_q_values[ddqn_dqn_dev_min_idx].item())
                    target_q_values_adj = target_q_values_ddqn

            if self.target == "proba_action_selection_dqn":
                
                all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                action_probas = th.softmax(all_next_q_values_offline, dim=1)
                actions = th.multinomial(action_probas, num_samples=1)

                next_q_values_ddqn = all_next_q_values_offline.gather(1, actions).reshape(-1, 1)
                target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                target_q_values_adj = target_q_values_ddqn

            if self.target == "proba_action_selection":

                all_next_q_values_online = self.q_net(replay_data.next_observations)
                all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                action_probas = th.softmax(all_next_q_values_online, dim=1)
                actions = th.multinomial(action_probas, num_samples=1)

                next_q_values_ddqn = all_next_q_values_offline.gather(1, actions).reshape(-1, 1)
                target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                target_q_values_adj = target_q_values_ddqn

            if self.target == "no_increase_on_misalignment_2":
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    online_down = target_q_values_online < current_q_values
                    offline_up = target_q_values_ddqn > current_q_values

                    misalignment = online_down & offline_up

                    target_q_values_adj = th.where(misalignment, current_q_values, target_q_values_ddqn)

            if self.target == "no_increase_on_misalignment":
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_down = target_q_values_online < current_q_values
                    ddqn_up = target_q_values_ddqn > current_q_values

                    misalignment = target_down & ddqn_up & action_change

                    target_q_values_adj = th.where(misalignment, current_q_values, target_q_values_ddqn)

            if self.target == "no_change_on_misalignment":
                
                all_next_q_values_online = self.q_net(replay_data.next_observations)
                all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                action_change = next_actions_online != next_actions_offline

                # Online Target
                next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                # DDQN target
                next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                # DQN Target
                next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                online_change = target_q_values_online - current_q_values
                ddqn_change = target_q_values_ddqn - current_q_values

                misalignment = ((online_change * ddqn_change) < 0) & action_change

                target_q_values_adj = th.where(misalignment, current_q_values, target_q_values_ddqn)

            if self.target == "misalignment_scaledown":
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_change = target_q_values_online - current_q_values
                    ddqn_change = target_q_values_ddqn - current_q_values

                    alignment = (target_change * ddqn_change) > 0

                    target_q_values_adj = th.where(alignment, current_q_values + ddqn_change, current_q_values + (ddqn_change / 2))

            if self.target == "decorrelate":

                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    noise = th.normal(mean=0.0, std=0.1, size=(batch_size, self.action_space.n))
                    all_next_q_values_hybrid = noise

                    next_actions_hybrid = all_next_q_values_hybrid.argmax(dim=1, keepdim=True)

                    next_q_values = all_next_q_values_offline.gather(1, next_actions_hybrid).reshape(-1, 1)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "stabilize_action_selection":

                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    all_next_q_values_hybrid = 0.5 * all_next_q_values_online + 0.5 * all_next_q_values_offline

                    next_actions_hybrid = all_next_q_values_hybrid.argmax(dim=1, keepdim=True)

                    next_q_values = all_next_q_values_offline.gather(1, next_actions_hybrid).reshape(-1, 1)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "backup_denoise":

                all_next_q_values = self.q_net(replay_data.next_observations)

                all_next_q_values_distorted = all_next_q_values.clone()

                for backup_policy in self.backup_policies:

                    noise = backup_policy.q_net(replay_data.next_observations)

                    all_next_q_values_distorted += noise

                next_actions = all_next_q_values_distorted.argmax(dim=1, keepdim=True)

                next_q_values = all_next_q_values.gather(1, next_actions).reshape(-1, 1)    

                target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "init_ensemble":

                iters = 5
                next_q_values_avg = 0

                all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                for _ in range(iters):

                    noise = th.normal(mean=0.0, std=0.1, size=(batch_size, self.action_space.n))

                    all_next_q_values_distorted = all_next_q_values_offline + noise

                    next_actions_distorted = all_next_q_values_distorted.argmax(dim=1, keepdim=True)

                    next_q_values = all_next_q_values_offline.gather(1, next_actions_distorted).reshape(-1, 1)                    

                    next_q_values_avg += ( (1 / iters) * next_q_values)

                target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_avg

            if self.target == "direction_specific":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)
                    action_change = next_actions_online != next_actions_offline

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn, _ = all_next_q_values_offline.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_q_values_adj = th.where(
                        action_change & (target_q_values_online > target_q_values_dqn),
                        target_q_values_dqn,
                        target_q_values_ddqn
                    )

            if self.target == "minimum":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    next_q_values_dqn = all_next_q_values_offline.gather(1, next_actions_offline).reshape(-1, 1)
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)

                    # offline_online_avg = (
                    #     next_q_values_online * relative_episodic_position + 
                    #     next_q_values_ddqn * (1-relative_episodic_position)
                    # )

                    next_q_values = th.minimum(next_q_values_dqn, next_q_values_online)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "minimum_late":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    next_q_values_dqn = all_next_q_values_offline.gather(1, next_actions_offline).reshape(-1, 1)
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)

                    offline_online_avg = (
                        next_q_values_online * relative_episodic_position + 
                        next_q_values_dqn * (1-relative_episodic_position)
                    )

                    next_q_values = th.minimum(next_q_values_dqn, offline_online_avg)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "minimum_avg":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    next_q_values_dqn = all_next_q_values_offline.gather(1, next_actions_offline).reshape(-1, 1)
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)

                    offline_online_avg = (
                        next_q_values_online * .5 + 
                        next_q_values_dqn * .5
                    )

                    next_q_values = th.minimum(next_q_values_dqn, offline_online_avg)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "minimum_dampened":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)

                    offline_online_avg = (
                        0.5 * next_q_values_ddqn +
                        0.5 * next_q_values_online
                    )

                    next_q_values = th.minimum(next_q_values_ddqn, offline_online_avg)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "minimum_dampened_late":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)

                    offline_online_avg = (
                        0.5 * next_q_values_ddqn +
                        0.5 * (
                            next_q_values_online * (1-relative_episodic_position) + 
                            next_q_values_ddqn * relative_episodic_position
                        )
                    )

                    next_q_values = th.minimum(next_q_values_ddqn, offline_online_avg)

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "ddqn_diff_fallback":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                    # If aligned, use DQN, else use average
                    no_change_mask = (next_actions_online == next_actions_offline).reshape(-1, 1)

                    next_q_values = all_next_q_values_offline.gather(1, next_actions_offline).reshape(-1, 1)
                    next_q_values_fallback = th.median(all_next_q_values_offline, dim=1).values.reshape(-1, 1)

                    next_q_values = th.where(no_change_mask, next_q_values, next_q_values_fallback)

                    # print(f"{next_actions_online[:3]=}")
                    # print(f"{next_actions_offline[:3]=}")
                    # print(f"{no_change_mask[:3]=}")
                    # print(f"{next_q_values[:3]=}")
                    # print(f"{next_q_values_fallback[:3]=}")
                    # print(f"{next_q_values[:3]=}")
                    # print("---")

                    target_q_values_adj = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            if self.target == "basic_blend":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)
                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    all_next_q_values_dqn = all_next_q_values_target
                    next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_q_values_adj = (
                        0.5 * target_q_values_ddqn + 
                        0.5 * (
                            target_q_values_dqn * relative_episodic_position + 
                            target_q_values_ddqn * (1-relative_episodic_position)
                        )
                    )

                    # target_q_values_adj = target_q_values_ddqn

            if self.target == "blend":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)
                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    frozen_target = target_q_values_ddqn
                    adaptive_target = target_q_values_online
                    
                    target_q_values_adj = (
                        .5 * frozen_target +
                        .5 * adaptive_target
                    )

                    # print(f"{all_next_q_values_online.shape=}")
                    # print(f"{next_actions_online.shape=}")
                    # print(f"{next_q_values_online.shape=}")

                    # print(f"{all_current_q_values_target.shape=}")
                    # print(f"{current_q_values_target.shape=}")
                    # print(f"{replay_data.dones=}")
                    
                    # print("---")

            if self.target == "blend2":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)
                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DQN Target
                    all_next_q_values_dqn = all_next_q_values_target
                    next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_q_values_adj = (
                        .5 * target_q_values_ddqn +
                        .5 * (
                            relative_episodic_position * target_q_values_ddqn +
                            (1-relative_episodic_position) * target_q_values_online
                        )
                    )

                    # target_q_values_adj = target_q_values_ddqn
   
            if self.target == "ol_late":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DQN Target
                    all_next_q_values_dqn = all_next_q_values_target
                    next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    adaptive_target = (
                        (1 - relative_episodic_position) * target_q_values_ddqn + 
                        relative_episodic_position * target_q_values_online
                    )

                    target_q_values_adj = (
                        0.5 * adaptive_target + 
                        0.5 * target_q_values_ddqn
                    )

            if self.target == "stale_avoidance":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DQN Target
                    # all_next_q_values_dqn = all_next_q_values_target
                    # next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    # next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    # target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    current_td = th.abs(current_q_values - target_q_values_ddqn)
                    # current_td_online = th.abs(current_q_values - target_q_values_online)

                    staleness_factor = 1 - th.clamp(current_td, min=0, max=1)

                    target_q_values_adj = (
                        staleness_factor * target_q_values_online + 
                        (1 -  staleness_factor) * target_q_values_ddqn
                    )

            if self.target == "min_online":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DQN Target
                    all_next_q_values_dqn = all_next_q_values_target
                    next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_q_values_adj = th.minimum(target_q_values_ddqn, target_q_values_online)

            if self.target == "dqn_ddqn_blend":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    all_next_q_values_dqn = all_next_q_values_target
                    next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn
                    
                    target_q_values_adj = (
                        relative_episodic_position * target_q_values_dqn +
                        (1 - relative_episodic_position) * target_q_values_ddqn
                    )

            if self.target == "dqn_ddqn_blend":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    all_next_q_values_dqn = all_next_q_values_target
                    next_q_values_dqn, _ = all_next_q_values_dqn.max(dim=1)
                    next_q_values_dqn = next_q_values_dqn.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn

                    target_q_values_adj = (
                        relative_episodic_position * target_q_values_dqn +
                        (1 - relative_episodic_position) * target_q_values_ddqn
                    )

            if self.target == "w_w_avg":
                
                with th.no_grad():
                    
                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_target = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # DDQN Target
                    next_q_values_ddqn = all_next_q_values_target.gather(1, next_actions_online)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    # DQN Target
                    next_q_values_dqn = all_next_q_values_target.max(dim=1).values.reshape(-1, 1)
                    target_q_values_dqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_dqn
                    
                    adaptive_target = (
                        relative_episodic_position * target_q_values_dqn +
                        (1-relative_episodic_position) * target_q_values_ddqn
                    )

                    target_q_values_adj = (
                        0.5 * target_q_values_ddqn +
                        0.5 * adaptive_target
                    )

            # loss = F.smooth_l1_loss(current_q_values, target_q_values_adj)

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

        self.train_counter +=1

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

