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

            if self.target == "loss_scale":

                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)
                    online_error_change =  th.abs(online_err) / (th.abs(online_err) + th.abs(new_online_err) + 1e-8)

                    online_error_change /= online_error_change.mean()

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * online_error_change
                loss = loss.mean()
        
            if self.target == "discard_prop_sample":

                oversampling_ratio = 2

                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size * oversampling_ratio, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)
                    online_error_change =  th.abs(online_err) / (th.abs(online_err) + th.abs(new_online_err) + 1e-8)
                    batch_idxes = th.multinomial(online_error_change.squeeze(), batch_size, replacement=False)

                    avg_ratio_before_discard = online_error_change.mean()
                    average_ratio_after_discard = online_error_change[batch_idxes].mean()
                    max_ratio_before_discard = online_error_change.max()
                    max_ratio_after_discard = online_error_change[batch_idxes].max()

                    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                    self.logger.record("train/loss", np.mean(losses))
                    self.logger.record("xustom/avg_ratio_before_discard", avg_ratio_before_discard.item(), exclude="tensorboard")
                    self.logger.record("xustom/average_ratio_after_discard", average_ratio_after_discard.item(), exclude="tensorboard")
                    self.logger.record("xustom/max_ratio_before_discard", max_ratio_before_discard.item(), exclude="tensorboard")
                    self.logger.record("xustom/lmax_ratio_after_discardoss", max_ratio_after_discard.item(), exclude="tensorboard")

                loss = F.smooth_l1_loss(current_q_values[batch_idxes], target_q_values_ddqn[batch_idxes])

            if self.target == "discard_prop_v2":

                oversampling_ratio = 2

                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size * oversampling_ratio, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                with th.no_grad():

                    all_next_q_values_online = self.q_net(replay_data.next_observations)
                    all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                    next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)

                    # Online Target
                    next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                    # DDQN target
                    next_q_values_ddqn = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)
                    target_q_values_ddqn = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_ddqn

                    ddqn_err = target_q_values_ddqn - current_q_values
                    online_err = target_q_values_online - current_q_values
                    new_online_err = target_q_values_online - (current_q_values + ddqn_err)
                    online_error_change =  th.abs(online_err) / (th.abs(online_err) + th.abs(new_online_err) + 1e-8)
                    _, batch_idxes = th.topk(online_error_change.squeeze(), k=batch_size)

                    avg_ratio_before_discard = online_error_change.mean()
                    average_ratio_after_discard = online_error_change[batch_idxes].mean()
                    max_ratio_before_discard = online_error_change.max()
                    max_ratio_after_discard = online_error_change[batch_idxes].max()

                    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                    self.logger.record("train/loss", np.mean(losses))
                    self.logger.record("xustom/avg_ratio_before_discard", avg_ratio_before_discard.item(), exclude="tensorboard")
                    self.logger.record("xustom/average_ratio_after_discard", average_ratio_after_discard.item(), exclude="tensorboard")
                    self.logger.record("xustom/max_ratio_before_discard", max_ratio_before_discard.item(), exclude="tensorboard")
                    self.logger.record("xustom/lmax_ratio_after_discardoss", max_ratio_after_discard.item(), exclude="tensorboard")

                loss = F.smooth_l1_loss(current_q_values[batch_idxes], target_q_values_ddqn[batch_idxes])

            """ Outdated approaches
            if self.target == "discard_prop":

                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size * 2, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

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

                    online_err_increase_raw = (th.abs(online_err) - th.abs(new_online_err)) / (th.abs(ddqn_err) + 1e-8)

                    _, batch_idxes = th.topk(online_err_increase_raw.squeeze(), k=batch_size)

                loss = F.smooth_l1_loss(current_q_values[batch_idxes], target_q_values_ddqn[batch_idxes])

            if self.target == "discard":

                to_discard = batch_size

                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size + to_discard, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

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

                    online_err_increase_raw = th.abs(online_err) - th.abs(new_online_err)

                    _, batch_idxes = th.topk(online_err_increase_raw.squeeze(), k=batch_size)

                loss = F.smooth_l1_loss(current_q_values[batch_idxes], target_q_values_ddqn[batch_idxes])

            if self.target == "importance_sampling":
                
                replay_data, sample_idxs, relative_episodic_position = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                relative_episodic_position = th.from_numpy(relative_episodic_position).to(self.device).unsqueeze(1).float()

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

                    # online_err_increase = th.abs(online_err) - th.abs(new_online_err)
                    online_err_increase_raw = th.abs(new_online_err) - th.abs(online_err)
                    online_err_increase = th.clamp(online_err_increase_raw, min=0, max=1)
                    downweight = online_err_increase / 2 

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

                loss = F.smooth_l1_loss(current_q_values, target_q_values_ddqn, reduction="none") * weights
                loss = loss.mean()

            """

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

