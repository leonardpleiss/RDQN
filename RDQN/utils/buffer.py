from stable_baselines3.common.buffers_custom_v2 import CustomPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSumProp, CustomPrioritizedReplayBufferCumSum2, CustomPrioritizedReplayBuffer, CustomPrioritizedReplayBufferCumSum3, CustomPrioritizedReplayBufferCumSum4, CustomPrioritizedReplayBufferCumSum5, CustomPropagatingPrioritizedReplayBuffer, CustomPropagatingPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSum6, CustomPrioritizedReplayBufferCumSum7
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers_custom import PrioritizedReplayBuffer, PrioritizedReplayBufferPropagating
from stable_baselines3.common.buffers_custom_v3 import ReaPER, R_UNI


def get_replay_buffer_config(buffer_name, debug_mode=True, check_frequency=100_000):

    if buffer_name == "R_UNI_a1":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .1,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a2":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .2,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a3":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .3,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a4":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .4,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a5":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .5,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": True,
    }
        
    
    elif buffer_name == "R_UNI_a1_mSN":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .1,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": True,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a2_mSN":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .2,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": True,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a3_mSN":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .3,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": True,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a4_mSN":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .4,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": True,
            "conservative_initial_reliabilities": True,
    }
        
    elif buffer_name == "R_UNI_a5_mSN":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .5,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": True,
            "conservative_initial_reliabilities": True,
    }
                
    elif buffer_name == "UNI":
        replay_buffer_class = ReplayBuffer
        replay_buffer_kwargs = {}

    elif buffer_name == "PER":
        replay_buffer_class = PrioritizedReplayBuffer
        replay_buffer_kwargs = {
            "alpha": .6
        }

    return replay_buffer_class, replay_buffer_kwargs