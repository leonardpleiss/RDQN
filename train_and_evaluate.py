from utils.storage import get_export_path, get_tb_storage_file_path
from utils.environment import get_environment_specific_settings, seed_everything 
from utils.buffer import get_replay_buffer_config
import cProfile
from pstats import Stats
from datetime import date
import sys
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.dqn.rdqn import RDQN

if __name__ == "__main__":
    ######################################## PARAMETERS ########################################

    # --------------------------------- # General Settings # --------------------------------- #

    save_weights = False
    profile = False
    device = "cuda"
    n_envs = 1
    trial_name = "0109_FULLRUN_v4"
    use_sb3_standard_params = False

    # ---------------------------------- # Trial Settings # ---------------------------------- #

    print(f"{sys.argv=}")

    if len(sys.argv) == 1:

        sys.argv.append("MinAtar/Breakout-v1") # ['NameThisGameNoFrameskip-v4', 'QbertNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'DoubleDunkNoFrameskip-v4', 'PhoenixNoFrameskip-v4']")
        sys.argv.append("PositionalReplayBuffer")
        sys.argv.append("RDQN")
        sys.argv.append("3")
        sys.argv.append("0")

    print(sys.argv)

    minatar_envs = [
        "MinAtar/Breakout-v1",
        "MinAtar/Freeway-v1",
        # "MinAtar/Seaquest-v1", # Takes forever
        "MinAtar/SpaceInvaders-v1",
        # "MinAtar/Asterix-v1", # Agent doesnt learn effectively
    ]

    environment_names = [sys.argv[1]] #minatar_envs # [sys.argv[1]] #minatar_envs# ["MinAtar/SpaceInvaders-v1"] # [sys.argv[1]] #minatar_envs # [sys.argv[1]] # minatar_envs # [sys.argv[1]] #minatar_envs #["MinAtar/Breakout-v1"] # ["LunarLander-v2", "CartPole-v1", "Acrobot-v1"]
    buffer_names = [sys.argv[2]] #["SelectiveReplayBuffer_02", "SelectiveReplayBuffer_04", "SelectiveReplayBuffer_06", "SelectiveReplayBuffer_08"] #[sys.argv[2]] # "R_UNI_a10"] #, "R_UNI_a8", "R_UNI_a6", "R_UNI_a4", "R_UNI_a2"] 
    model_names = [sys.argv[3]] #"RDQN"] # [sys.argv[3]] # ["RDQN", "RDQN", "RDQN", "DDQN"] # [sys.argv[3]]
    iterations_per_env = int(sys.argv[4])
    starting_seed = int(sys.argv[5])

    targets = ["discard_OSR2_DQN_errscaled"] #["discard_sample_OSR2_DQN", "discard_OSR2_DQN", "loss_scale_DQN", ""] # ["loss_scale", "discard_prop_sample", "discard_prop_v2", ""]

    ##############################################################################################
     
    trial_start_date = date.today().strftime("%Y%m%d")

    for environment_name in environment_names:
        for iteration in range(starting_seed, iterations_per_env+starting_seed):
            for buffer_name in buffer_names:
                for model_idx, model_name in enumerate(model_names):

                    target = targets[model_idx]

                    if model_name in ["DDQN", "DQN"]:
                        buffer_name = "UNI"
                    elif model_name in ["RDQN"]:
                        buffer_name = "PositionalReplayBuffer"

                    print(f"seed: {iteration}")
                    seed_everything(iteration)

                    replay_buffer_class, replay_buffer_kwargs = get_replay_buffer_config(buffer_name=buffer_name)
                    export_suffix = f"{trial_start_date}_{trial_name}_{buffer_name}"

                    if "alpha2" in list(replay_buffer_kwargs.keys()):
                        alpha2 = replay_buffer_kwargs["alpha2"]

                    tb_log_path = get_tb_storage_file_path(environment_name, replay_buffer_class, model_name)
                    trial_result_folder_path = get_export_path(model_name, environment_name, replay_buffer_class.__name__)

                    model_class, env, eval_env, dqn_policy, max_timesteps, reward_threshold, num_evals, \
                    callback_on_new_best, learning_rate, learning_starts, n_eval_episodes, \
                    exploration_initial_eps, exploration_fraction, exploration_final_eps, \
                    batch_size, buffer_size, policy_kwargs, max_grad_norm, train_freq, \
                    target_update_interval, gradient_steps, gamma, eval_exploration_fraction, progress_bar = \
                    get_environment_specific_settings(model_name, environment_name, n_envs=n_envs, seed=iteration, use_sb3_standard_params=use_sb3_standard_params)

                    export_suffix += f"_{batch_size}bs_{iteration}seed_{int(num_evals)}evalfreq_{n_envs}"

                    if model_name == "RDQN":
                        export_suffix += f"_{target}"

                    model = model_class(

                        # General settings
                        env=env,
                        verbose=1,
                        seed=iteration,
                        device=device,

                        # Hyperparameters
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        learning_starts=learning_starts,
                        exploration_initial_eps=exploration_initial_eps,
                        exploration_fraction=exploration_fraction,
                        exploration_final_eps=exploration_final_eps,
                        policy=dqn_policy,
                        policy_kwargs=policy_kwargs,
                        max_grad_norm=max_grad_norm,
                        train_freq=train_freq,
                        target_update_interval=target_update_interval,
                        gradient_steps=gradient_steps,
                        gamma=gamma,

                        # Tensorboard settings
                        tensorboard_log = tb_log_path + export_suffix + "/",

                        # Replay Buffer Settings
                        replay_buffer_class=replay_buffer_class,
                        replay_buffer_kwargs=replay_buffer_kwargs,
                        replay_buffer_log_path=trial_result_folder_path,
                    )

                    if model_name == "RDQN":
                        model.target = target
                        print(f"{target} is used for updating {model_class}.")

                    eval_callback = EvalCallback(eval_env, eval_freq=max_timesteps/num_evals/n_envs, 
                                                callback_on_new_best=callback_on_new_best,
                                                verbose=1, n_eval_episodes=n_eval_episodes,
                                                exploration_fraction=eval_exploration_fraction)

                    if profile:
                        pr = cProfile.Profile()
                        pr.enable()

                    model.learn(total_timesteps=max_timesteps, log_interval=max_timesteps/num_evals, progress_bar=progress_bar, callback=eval_callback)

                    if profile:
                        pr.disable()
                        stats = Stats(pr)
                        stats.sort_stats('tottime').print_stats(200)

                    if save_weights:
                        model.save(trial_result_folder_path + "weights")