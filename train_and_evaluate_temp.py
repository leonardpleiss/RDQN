from utils.storage import get_export_path, get_tb_storage_file_path
from utils.environment import get_environment_specific_settings, seed_everything 
from utils.buffer import get_replay_buffer_config
import cProfile
from pstats import Stats
from datetime import date
import sys
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat, CSVOutputFormat, TensorBoardOutputFormat
import os
import ast


if __name__ == "__main__":
    ######################################## PARAMETERS ########################################

    # --------------------------------- # General Settings # --------------------------------- #

    save_weights = False
    profile = False
    device = "cuda"
    n_envs = 1
    debug_mode = False
    allow_duplicates_in_batch = True
    full_check_frequency = 1_000
    trial_name = "1807_RDQN_BIG_TRIAL" # 1607_RDQN_02 --> div by mean, was good
    use_sb3_standard_params = False
    RWTH_cluster = False

    # ---------------------------------- # Trial Settings # ---------------------------------- #

    print(f"{sys.argv=}")

    if len(sys.argv) == 1:

        sys.argv.append("LunarLander-v2") # ['NameThisGameNoFrameskip-v4', 'QbertNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'DoubleDunkNoFrameskip-v4', 'PhoenixNoFrameskip-v4']")
        sys.argv.append("R_UNI_a3")
        sys.argv.append("RDQN")
        sys.argv.append("20")
        sys.argv.append("0")

    print(sys.argv)

    if RWTH_cluster:
        environment_names = ast.literal_eval(sys.argv[1]) # Available: "CartPole-v1", "LunarLander-v2", "FrozenLake-v1", "Acrobot-v1", "CliffWalking-v0", "Taxi-v3", "BlindCliffwalk-v0", "PongNoFrameskip-v4"
    else:
        environment_names = [sys.argv[1]]

    environment_names = ["CartPole-v1", "Acrobot-v1", "LunarLander-v2"] # 
    buffer_names = ["R_UNI_a1_mSN", "R_UNI_a2_mSN", "R_UNI_a3_mSN", "R_UNI_a4_mSN", "R_UNI_a5_mSN", "R_UNI_a1", "R_UNI_a2", "R_UNI_a3", "R_UNI_a4", "R_UNI_a5"] # ["R_UNI_a3"] #, "R_UNI_a125", "R_UNI_a15", "R_UNI_a2", "R_UNI_a3"] # 
    model_names = [sys.argv[3]]
    iterations_per_env = int(sys.argv[4])
    starting_seed = int(sys.argv[5])

    ##############################################################################################
    
    trial_start_date = date.today().strftime("%Y%m%d")

    for environment_name in environment_names:
        for iteration in range(starting_seed, iterations_per_env+starting_seed):
            for buffer_name in buffer_names:
                for model_name in model_names:
                    print(f"seed: {iteration}")

                    # FIX - MAYBE DELETE!!!
                    if (buffer_name == "UNI") & (model_name == "RDQN"):
                        model_name = "DQN"
                    
                    elif (buffer_name == "UNI") & (model_name == "RDDQN"):
                        model_name = "DDQN"

                    seed_everything(iteration)

                    replay_buffer_class, replay_buffer_kwargs = get_replay_buffer_config(buffer_name=buffer_name, debug_mode=debug_mode, check_frequency=full_check_frequency)
                    export_suffix = f"{trial_start_date}_{trial_name}_{buffer_name}"

                    if replay_buffer_kwargs:
                        alpha2 = replay_buffer_kwargs["alpha2"]

                    tb_log_path = get_tb_storage_file_path(environment_name, replay_buffer_class, model_name)
                    trial_result_folder_path = get_export_path(model_name, environment_name, replay_buffer_class)

                    model_class, env, eval_env, dqn_policy, max_timesteps, reward_threshold, num_evals, \
                    callback_on_new_best, learning_rate, learning_starts, n_eval_episodes, \
                    exploration_initial_eps, exploration_fraction, exploration_final_eps, \
                    batch_size, buffer_size, policy_kwargs, max_grad_norm, train_freq, \
                    target_update_interval, gradient_steps, gamma, eval_exploration_fraction, progress_bar = \
                    get_environment_specific_settings(model_name, environment_name, n_envs=n_envs, seed=iteration, use_sb3_standard_params=use_sb3_standard_params)

                    export_suffix += f"_{batch_size}bs_{iteration}seed_{int(num_evals)}evalfreq_{n_envs}env"

                    print(export_suffix)

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

                    # Get custom logger if running on RWTH cluster
                    if RWTH_cluster:
                        log_path = str(f"logs/{trial_name}/{environment_name}")  # or your custom path
                        os.makedirs(log_path, exist_ok=True)

                        # Custom file name for CSV
                        csv_file = os.path.join(log_path, f"{trial_name}_{environment_name}.csv")
                        stdout_logger = HumanOutputFormat(sys.stdout)

                        # Create a custom Logger instance
                        new_logger = Logger(
                            folder=log_path,
                            output_formats=[
                                stdout_logger,
                                CSVOutputFormat(csv_file),
                                TensorBoardOutputFormat(log_path),
                            ]
                        )
                        model.set_logger(new_logger)

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