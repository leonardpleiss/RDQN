from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
import random
import torch
import os
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
import torch.optim as optim
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.ddqn import DDQN
from stable_baselines3.dqn.rdqn import RDQN
from stable_baselines3.dqn.rddqn import RDDQN

atari_5 = [
    "NameThisGameNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
]

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
    "ZaxxonNoFrameskip-v4"
]

def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_environment_specific_settings(model_name, environment_name, n_envs:int=1, seed:int=0, use_sb3_standard_params=False):

    assert n_envs == 1

    if model_name == "DQN":
        model_class = DQN
    elif model_name == "DDQN":
        model_class = DDQN
    elif model_name == "RDQN":
        model_class = RDQN
    elif model_name == "RDDQN":
        model_class = RDDQN
    else:
        raise ValueError(f"Model name not supported: {model_name}")

    # Set standard settings
    env = make_vec_env(environment_name, n_envs=n_envs, seed=seed)
    eval_env = make_vec_env(environment_name, n_envs=1, seed=seed)

    # General settings & default parameters - Parameters will be overwritten subsequently if a given environment requires a different specification.
    dqn_policy = "MlpPolicy"
    reward_threshold = gym.spec(environment_name).reward_threshold
    num_evals = 100
    callback_on_new_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    learning_rate=0.0001
    learning_starts = 100
    n_eval_episodes = 5
    exploration_initial_eps = 1.
    exploration_fraction=.1
    exploration_final_eps = .05
    buffer_size = 1_000_000
    batch_size = 32
    policy_kwargs = None
    gradient_steps = 1
    max_grad_norm = 10.
    target_update_interval = 10_000
    train_freq = 4
    progress_bar = True
    gamma = 0.99
    eval_exploration_fraction = .0

    if environment_name == "CartPole-v1":

        num_timesteps = 50000
        learning_rate=2.3e-3
        learning_starts = 1000

        if not use_sb3_standard_params:
            
            batch_size=64
            buffer_size=100000
            learning_starts=1000
            target_update_interval=10
            train_freq=256
            gradient_steps=128
            exploration_fraction=0.16
            exploration_final_eps=0.04
            policy_kwargs=dict(net_arch=[256, 256])
            num_evals = 100

    elif environment_name == "CartPole-v0":

        num_timesteps = 1_010_000
        learning_rate=2.3e-3
        learning_starts = 1_000_000

        if not use_sb3_standard_params:
            
            batch_size=32
            buffer_size=1_000_000
            target_update_interval=10
            train_freq=256
            gradient_steps=128
            exploration_fraction=0.16
            exploration_final_eps=0.04
            policy_kwargs=dict(net_arch=[256, 256])
            num_evals = 1

    elif environment_name == "LunarLander-v2":

        num_timesteps = 1e5
        learning_rate = 6.3e-4
        learning_starts = 1000

        if not use_sb3_standard_params:
            
            batch_size = 128
            buffer_size = 50000
            learning_starts = 1000
            target_update_interval = 250
            train_freq = 4
            gradient_steps = -1
            exploration_fraction = 0.12
            exploration_final_eps = 0.1
            policy_kwargs = dict(net_arch=[256, 256])
            num_evals = 100
            n_eval_episodes = 1

    elif environment_name == "MountainCar-v0":
        
        num_timesteps = 1.2e5
        learning_rate = 4e-3
        learning_starts = 1000

        if not use_sb3_standard_params:
            batch_size = 128
            buffer_size = 10000
            learning_starts = 1000
            gamma = 0.98
            target_update_interval = 600
            train_freq = 16
            gradient_steps = 8
            exploration_fraction = 0.2
            exploration_final_eps = 0.07
            policy_kwargs = dict(net_arch=[256, 256])
            num_evals = 100

    elif environment_name == "Acrobot-v1":

        num_timesteps = 1e5
        learning_rate = 6.3e-4
        learning_starts = 1000

        if not use_sb3_standard_params:

            batch_size = 128
            buffer_size = 50000
            learning_starts = 1000
            target_update_interval = 250
            train_freq = 4
            gradient_steps = -1
            exploration_fraction = 0.12
            exploration_final_eps: 0.1
            policy_kwargs = dict(net_arch=[256, 256])
            num_evals = 100

    # Overwrite standard settings if needed & specify env-specific parameters
    elif environment_name in all_atari_games:

        env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)

        eval_env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        eval_env = VecTransposeImage(eval_env)

        dqn_policy = "CnnPolicy"
        num_timesteps = 50_000_000
        reward_threshold = np.inf
        num_evals = 200
        learning_starts = 50_000 # Not mentioned for DDQN or PER. Set according to DQN (Mnih et al., 2015)
        n_eval_episodes=1 # Set according to PER (Schaul et al, 2015) for DDQN 
        callback_on_new_best = None
        learning_rate=0.00025 # For PER, set to 0.00025 / 4 (Schaul et al, 2015) for DDQN 
        exploration_initial_eps=1. # Set according to DDQN paper (van Hasselt, 2015 not mentioned in PER. 
        exploration_fraction=.02 # Set according to DDQN paper (van Hasselt, 2015 not mentioned in PER. 
        train_freq = 4
        progress_bar = False
        gradient_steps = 1
        max_grad_norm = np.inf # Gradient clipping is not mentioned in any paper. Not set in DeepMind rlzoo for DQN, DDQN or PER.
        gamma = 0.99
        policy_kwargs = dict(
            optimizer_class=optim.RMSprop,
            optimizer_kwargs=dict(alpha=0.95,
                                  eps=0.01,
                                  momentum=0.95, # Set according to original DDQN for RL paper (van Hasselt, 2015)
                                  centered=True))
        
        if model_name in ["DQN", "RDQN"]:
            print("DQN parameters loaded")
            exploration_final_eps=.1 # Set according to DDQN paper (van Hasselt, 2015 not mentioned in PER. 
            target_update_interval=10_000 # Set according to DDQN paper (van Hasselt, 2015).
            eval_exploration_fraction = .05

        elif model_name in ["DDQN", "RDDQN"]:
            print("DDQN parameters loaded")
            exploration_final_eps=.01 # Set according to DDQN paper (van Hasselt, 2015 not mentioned in PER. 
            target_update_interval=30_000 # Set according to DDQN paper (van Hasselt, 2015).
            eval_exploration_fraction = .001

        else:
            raise ValueError(f"Specify either DDQN or DQN as model - currently specified: {model_name}.")

    elif environment_name == "BlindCliffwalk-v0":
        num_timesteps = 20_000

    elif environment_name == "Taxi-v3":
        num_timesteps = 1_000_000

    elif environment_name == "CarRacing-v2":
        num_timesteps = 3_000_000

    elif environment_name == "CliffWalking-v0":
        num_timesteps = 300_000

    elif environment_name == "FrozenLake-v1":
        num_timesteps = 100_000
        env = make_vec_env(environment_name, n_envs=n_envs, seed=seed, env_kwargs={"is_slippery": False,
                                                                                   "map_name": "4x4"})

    elif environment_name == "Blackjack-v1":
        num_timesteps = 300_000

    else:
        raise NotImplementedError("This environment is currently not supported")

    try:
        print(env)
        print(env.envs[0].env)
        print(env.envs[0].env.env)
        print(env.envs[0].env.env.env)
        print(env.envs[0].env.env.env.env)
        print(dir(env.envs[0].env.env.env.env))
        print(env.envs[0].env.env.env.env.spec)
    except:
        pass

    print(f"{reward_threshold, learning_starts, batch_size=}")


    return  model_class, env, eval_env, dqn_policy, num_timesteps, reward_threshold, num_evals, callback_on_new_best, learning_rate, \
            learning_starts, n_eval_episodes, exploration_initial_eps, exploration_fraction, exploration_final_eps, batch_size, buffer_size, \
            policy_kwargs, max_grad_norm, train_freq, target_update_interval, gradient_steps, gamma, eval_exploration_fraction, progress_bar