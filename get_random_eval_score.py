import os
from utils.storage import get_export_path, get_tb_storage_file_path
from utils.environment import get_environment_specific_settings, seed_everything 
from utils.buffer import get_replay_buffer_config
from stable_baselines3.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import cProfile
from pstats import Stats
from datetime import date
import sys
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import copy
import numpy as np
import sys
import stable_baselines3

if __name__ == "__main__":
    ######################################## PARAMETERS ########################################

    # --------------------------------- # General Settings # --------------------------------- #

    buffer_size = None # Default: None
    num_logs = 1 # Default: 10
    save_weights = False
    profile = False
    batch_size = None # Default: None
    device = "cuda"
    n_envs = 1
    debug_mode = False

    # ---------------------------------- # Trial Settings # ---------------------------------- #

    print(f"{sys.argv=}")

    if len(sys.argv) == 1:
        sys.argv.append("PongNoFrameskip-v4")
        sys.argv.append("TPER2")
        sys.argv.append("1")

    environment_names = [sys.argv[1]] # Available: "CartPole-v1", "LunarLander-v2", "FrozenLake-v1", "Acrobot-v1", "CliffWalking-v0", "Taxi-v3", "BlindCliffwalk-v0", "PongNoFrameskip-v4"
    buffer_names =  [sys.argv[2]] # Available: "TPER", "PER", "UNI", "PPER", "TPPER"
    trial_name = "tst"
    iterations_per_env = int(sys.argv[3])
    allow_duplicates_in_batch = True
    full_check_frequency = 100

    ##############################################################################################

    random_scores = {}

    if batch_size is None:
        batch_size = 32
    if buffer_size is None:
        buffer_size = 1_000_000
    
    trial_start_date = date.today().strftime("%Y%m%d")

    all_atari_games = [
        "AlienNoFrameskip-v4",
        "AmidarNoFrameskip-v4",
        "AssaultNoFrameskip-v4",
        "AsterixNoFrameskip-v4",
        "AsteroidsNoFrameskip-v4",
        "AtlantisNoFrameskip-v4",
        "BankHeistNoFrameskip-v4",
        "BattleZoneNoFrameskip-v4",
        "BeamRiderNoFrameskip-v4",
        "BerzerkNoFrameskip-v4",
        "BowlingNoFrameskip-v4",
        "BoxingNoFrameskip-v4",
        "BreakoutNoFrameskip-v4",
        "CentipedeNoFrameskip-v4",
        "ChopperCommandNoFrameskip-v4",
        "CrazyClimberNoFrameskip-v4",
        "DefenderNoFrameskip-v4",
        "DemonAttackNoFrameskip-v4",
        "DoubleDunkNoFrameskip-v4",
        "EnduroNoFrameskip-v4",
        "FishingDerbyNoFrameskip-v4",
        "FreewayNoFrameskip-v4",
        "FrostbiteNoFrameskip-v4",
        "GopherNoFrameskip-v4",
        "GravitarNoFrameskip-v4",
        "HeroNoFrameskip-v4",
        "IceHockeyNoFrameskip-v4",
        "JamesbondNoFrameskip-v4",
        "JourneyEscapeNoFrameskip-v4",
        "KangarooNoFrameskip-v4",
        "KrullNoFrameskip-v4",
        "KungFuMasterNoFrameskip-v4",
        "MontezumaRevengeNoFrameskip-v4",
        "MsPacmanNoFrameskip-v4",
        "NameThisGameNoFrameskip-v4",
        "PhoenixNoFrameskip-v4",
        "PitfallNoFrameskip-v4",
        "PongNoFrameskip-v4",
        "PrivateEyeNoFrameskip-v4",
        "QbertNoFrameskip-v4",
        "RiverraidNoFrameskip-v4",
        "RoadRunnerNoFrameskip-v4",
        "RobotankNoFrameskip-v4",
        "SeaquestNoFrameskip-v4",
        "SkiingNoFrameskip-v4",
        "SolarisNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4",
        "StarGunnerNoFrameskip-v4",
        "TennisNoFrameskip-v4",
        "TimePilotNoFrameskip-v4",
        "TutankhamNoFrameskip-v4",
        "UpNDownNoFrameskip-v4",
        "VentureNoFrameskip-v4",
        "VideoPinballNoFrameskip-v4",
        "WizardOfWorNoFrameskip-v4",
        "YarsRevengeNoFrameskip-v4",
        "ZaxxonNoFrameskip-v4",
    ]

    for environment_name in ["CartPole-v1", "MountainCar-v0", "LunarLander-v2", "Acrobot-v1"]:
        for iteration in range(iterations_per_env):
            for buffer_name in buffer_names:

                model_name = "DDQN"
                print(f"{iteration=}")

                print(f"{iteration=}")
                seed_everything(iteration)

                replay_buffer_class, replay_buffer_kwargs = get_replay_buffer_config(buffer_name=buffer_name, debug_mode=debug_mode, check_frequency=full_check_frequency)
                export_suffix = f"{trial_start_date}_{trial_name}_{buffer_name}"

                if replay_buffer_kwargs:
                    alpha = replay_buffer_kwargs["alpha"]
                    export_suffix += f"_{int(alpha*10)}a"

                tb_log_path = get_tb_storage_file_path(environment_name, replay_buffer_class, model_name)
                trial_result_folder_path = get_export_path(model_name, environment_name, replay_buffer_class)

                model_class, env, eval_env, dqn_policy, max_timesteps, reward_threshold, num_evals, \
                callback_on_new_best, learning_rate, learning_starts, n_eval_episodes, \
                exploration_initial_eps, exploration_fraction, exploration_final_eps, \
                batch_size, buffer_size, policy_kwargs, max_grad_norm, train_freq, \
                target_update_interval, gradient_steps, gamma, progress_bar = \
                get_environment_specific_settings(model_name, environment_name, n_envs=n_envs, seed=iteration)

                export_suffix += f"_{batch_size}bs_{iteration}seed_{int(num_evals)}evalfreq_{n_envs}env"

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

                eval_callback = EvalCallback(eval_env, eval_freq=max_timesteps/num_evals/n_envs, 
                                            callback_on_new_best=callback_on_new_best,
                                            verbose=1, n_eval_episodes=n_eval_episodes)

                eval_env.metadata['render_fps'] = 60

                mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, random_actions=True, render=True)
                print(f"{environment_name}: Avg. Score: {(mean_reward):.2f}")

                random_scores[environment_name] = mean_reward

    print(random_scores)
    # os.system('say "Programmausführung abgeschlossen"')