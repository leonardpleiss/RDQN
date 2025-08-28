from stable_baselines3.common.buffers_custom_v2 import CustomPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSumProp, CustomPrioritizedReplayBufferCumSum2, CustomPrioritizedReplayBuffer, CustomPrioritizedReplayBufferCumSum3, CustomPrioritizedReplayBufferCumSum4, CustomPrioritizedReplayBufferCumSum5, CustomPropagatingPrioritizedReplayBuffer, CustomPropagatingPrioritizedReplayBufferCumSum, CustomPrioritizedReplayBufferCumSum6, CustomPrioritizedReplayBufferCumSum7
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers_custom import PrioritizedReplayBuffer, PrioritizedReplayBufferPropagating
from stable_baselines3.common.buffers_custom_v3 import R_UNI
from stable_baselines3.common.buffers_custom_v4 import DR_UNI
from stable_baselines3.common.buffers_custom_v5 import PositionalReplayBuffer
from stable_baselines3.common.buffers_custom_v9 import SelectiveReplayBuffer
from stable_baselines3.common.buffers_custom_v10 import ForceIncludeReplayBuffer


def get_replay_buffer_config(buffer_name, check_frequency=100_000):
    
    if buffer_name == "PositionalReplayBuffer":
        replay_buffer_class = PositionalReplayBuffer
        replay_buffer_kwargs = {
    }
        
    if buffer_name == "ForceIncludeReplayBuffer":
        replay_buffer_class = ForceIncludeReplayBuffer
        replay_buffer_kwargs = {
    }

    if buffer_name == "SelectiveReplayBuffer":
        replay_buffer_class = SelectiveReplayBuffer
        replay_buffer_kwargs = {
    }
        
    if buffer_name == "SelectiveReplayBuffer_02":
        replay_buffer_class = SelectiveReplayBuffer
        replay_buffer_kwargs = {
            "signal_ratio": .2,
    }
        
    if buffer_name == "SelectiveReplayBuffer_04":
        replay_buffer_class = SelectiveReplayBuffer
        replay_buffer_kwargs = {
            "signal_ratio": .4,
    }
        
    if buffer_name == "SelectiveReplayBuffer_06":
        replay_buffer_class = SelectiveReplayBuffer
        replay_buffer_kwargs = {
            "signal_ratio": .6,
    }
        
    if buffer_name == "SelectiveReplayBuffer_08":
        replay_buffer_class = SelectiveReplayBuffer
        replay_buffer_kwargs = {
             "signal_ratio": .8,
    }
        
    if buffer_name == "DR_UNI_a10":
        replay_buffer_class = DR_UNI
        replay_buffer_kwargs = {
            "alpha2": 1.,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
    }
        
    if buffer_name == "R_UNI_a10":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha2": 1.,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
    }
    elif buffer_name == "R_UNI_a8":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha2": .8,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
    }
        
    elif buffer_name == "R_UNI_a6":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha2": .6,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
    }
        
    elif buffer_name == "R_UNI_a4":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha2": .4,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
    }
        
    elif buffer_name == "R_UNI_a2":
        replay_buffer_class = R_UNI
        replay_buffer_kwargs = {
            "alpha2": .2,
            "handle_timeout_termination": False,
            "check_frequency": check_frequency,
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