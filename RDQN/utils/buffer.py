from stable_baselines3.common.buffers_custom_v2 import CustomPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSumProp, CustomPrioritizedReplayBufferCumSum2, CustomPrioritizedReplayBuffer, CustomPrioritizedReplayBufferCumSum3, CustomPrioritizedReplayBufferCumSum4, CustomPrioritizedReplayBufferCumSum5, CustomPropagatingPrioritizedReplayBuffer, CustomPropagatingPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSum6, CustomPrioritizedReplayBufferCumSum7
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers_custom import PrioritizedReplayBuffer, PrioritizedReplayBufferPropagating
from stable_baselines3.common.buffers_custom_v3 import ReaPER, R_UNI


def get_replay_buffer_config(buffer_name, debug_mode=True, check_frequency=100_000):

    if buffer_name == "R_UNI_r0":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .0,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
        
    elif buffer_name == "R_UNI_NoSumUpdate":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": 1.,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
            "update_sums": False
    }
        
    elif buffer_name == "R_UNI_r2_NoSumUpdate":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .2,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
            "update_sums": False
    }
        
    elif buffer_name == "R_UNI_r1":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .1,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
        
    elif buffer_name == "R_UNI_r1_RR":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .1,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": True,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
        
    elif buffer_name == "R_UNI_r2":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .2,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }

    elif buffer_name == "R_UNI_r2_RR":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .2,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": True,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
        
    elif buffer_name == "R_UNI_r3":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .3,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
            
    elif buffer_name == "R_UNI_r3_RR":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .3,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": True,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
        
    elif buffer_name == "R_UNI_r4":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .4,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
    }
        
    elif buffer_name == "R_UNI_r5":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha": 1.,
            "alpha2": .5,
            "debug_mode": debug_mode,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
            "use_reward_ratios": False,
            "max_sum_normalization": False,
            "conservative_initial_reliabilities": False,
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