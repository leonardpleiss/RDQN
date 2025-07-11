import numpy as np
import copy
import sys
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

####################################################### PARAMETERS #######################################################

# Experiment parameters
methods = ["PER", "ReaPER"] # Available are "UNI", "PER", "ReaPER"
approximators = ["TQL"]# , "LFA", "DQN"]#, "DQN"] # "DQN" or "LFA" or "TQL"
max_episode_lengths = [16]
cutoff_timesteps = 1_500_000
iterations = 10
num_training_steps = 100_000

# Run parameters
init_max = .1
alpha = 1.
alpha2 = 1.
use_IS = False
tol = 1e-3
seed = 0
lr = .25
stochastic = False

reward_type = "random" # "single" or "random"
use_discount_factor = True

##########################################################################################################################
for stochastic in [False]:

    for approximator in approximators:

        result_dict = {
            "episode_length": [],
            "num_transitions": [],
            "UNI_mean": [],
            "UNI_sd": [],
            "UNI_min": [],
            "UNI_max": [],
            "PER_mean": [],
            "PER_sd": [],
            "PER_min": [],
            "PER_max": [],
            "ReaPER_mean": [],
            "ReaPER_sd": [],
            "ReaPER_min": [],
            "ReaPER_max": [],
        }

        for episode_length in max_episode_lengths:
            
            if use_discount_factor:
                discount_factor = 1. - (1/episode_length)
            else:
                discount_factor = 1

            np.random.seed(seed)

            num_final_states = 2 ** episode_length
            num_nonfinal_states = num_final_states - 1 

            if reward_type == "single":
                final_rewards = np.zeros((num_final_states))
                final_rewards[0] = 1
            elif reward_type == "random":
                final_rewards = np.random.choice(np.arange(num_final_states), num_final_states, replace=False) # Random high
            
            action_plans = np.array(list(itertools.product([0, 1], repeat=episode_length)))
            states, states_encoded, next_states, next_states_encoded, actions, episodes, rewards, dones = [], [], [], [], [], [], [], []

            target_q_values = np.hstack([(np.max(final_rewards.reshape(2**i, -1), axis=1)).flatten() * (discount_factor ** (episode_length - i)) for i in range(1,episode_length+1)]).reshape(-1, 2)

            for episode_idx, action_plan in enumerate(action_plans):

                state_encoded = np.zeros(episode_length * 2)
                state = 0

                for num_action, action in enumerate(action_plan):

                    states_encoded.append(state_encoded)
                    states.append(state)

                    next_state_encoded = copy.copy(state_encoded)
                    next_state_encoded[num_action*2+action] = 1
                    
                    next_state = copy.copy(state)
                    next_state = next_state * 2 + 1 + action

                    done = True if ((num_action+1) == episode_length) else False

                    if done:
                        reward = final_rewards[episode_idx]

                    elif not done:
                        reward = 0

                    dones.append(done)
                    actions.append(action)
                    next_states_encoded.append(state_encoded)
                    episodes.append(episode_idx+1)
                    rewards.append(reward)
                    next_states.append(next_state)

                    state_encoded = copy.copy(next_state_encoded)
                    state = copy.copy(next_state)

            print(f"Episode length: {episode_length}")
            print(f"Number of transitions: {len(states)}")
            print(f"Number of unique states: {len(target_q_values)}")
            result_dict["episode_length"].append(episode_length) 
            result_dict["num_transitions"].append(len(states)) 

            def get_rel_weights(td_errors, episodes, alpha2):

                # Convert inputs to numpy arrays
                td_errors = np.asarray(td_errors, dtype=np.float64)
                episodes = np.asarray(episodes)

                # Identify unique episodes and their starting indices
                unique_episodes, inverse_indices, episode_counts = np.unique(episodes, return_inverse=True, return_counts=True)

                # Compute per-episode cumulative sums
                cumsum_all = np.cumsum(td_errors)
                episode_start_indices = np.r_[0, np.cumsum(episode_counts)[:-1]]  # Start index of each episode

                # Compute td_cums: cumulative sum per episode
                episode_start_cumsum = cumsum_all[(episode_start_indices-1)[1:]]
                episode_start_cumsum = np.insert(episode_start_cumsum, 0, 0)

                td_cums = cumsum_all - episode_start_cumsum[inverse_indices]

                # Compute td_sums: total sum for each episode (mapped to original shape)
                total_sums = np.bincount(inverse_indices, weights=td_errors)
                td_sums = total_sums[inverse_indices]

                # Compute subsequent TD errors
                subsequent_tds = td_sums - td_cums

                # max_subsequent_td = np.max(subsequent_tds)
                max_subsequent_td = np.max(subsequent_tds)
                # print(f"{max_subsequent_td=}")

                # Compute relative weights
                rel_weight = (1 - (subsequent_tds / max_subsequent_td)) ** alpha2

                return rel_weight

            def get_sampling_idx(method, td_errors, episodes, alpha, use_IS):

                num_transitions = np.arange(len(states))

                if method == "UNI":
                    sampling_probas = np.ones_like(num_transitions) / len(num_transitions)
                    idx = np.random.choice(num_transitions)
                    
                elif method == "PER":
                    sampling_weights = td_errors ** alpha
                    sampling_probas = sampling_weights / sampling_weights.sum()

                    if stochastic:
                        idx = np.random.choice(num_transitions, p=sampling_probas)
                    else:
                        idx = np.argmax(sampling_probas)

                elif method == "ReaPER":
                    rel_weights = get_rel_weights(td_errors, episodes, alpha2)
                    sampling_weights = (td_errors ** alpha) * rel_weights
                    sampling_probas = sampling_weights / sampling_weights.sum()

                    if stochastic:
                        idx = np.random.choice(num_transitions, p=sampling_probas)
                    else:
                        idx = np.argmax(sampling_probas)

                # IS weights
                if use_IS:
                    beta = .6
                    p_min = sampling_probas.min()
                    max_weight = (p_min * len(num_transitions)) ** (-beta)
                    IS_weight = (sampling_probas[idx] * len(num_transitions)) ** (-beta) / max_weight
                    assert (IS_weight > 0).any(), f"{method}, {sampling_probas}, {sampling_probas[idx]}, {len(num_transitions)}, {(-beta)}, {max_weight}"

                else:
                    IS_weight = 1

                return idx, states[idx], actions[idx], IS_weight, sampling_probas

            # Define the Q-learning agent model in PyTorch
            class QLearningAgent(nn.Module):
                def __init__(self):
                    super(QLearningAgent, self).__init__()
                    # Define the model layers

                    if approximator == "LFA":
                        self.model = nn.Sequential(
                            nn.Linear(num_nonfinal_states+1, 2)  # Output layer
                        )

                    if approximator == "DQN":
                        self.model = nn.Sequential(
                            nn.Linear(num_nonfinal_states+1, num_nonfinal_states//2),
                            nn.ReLU(),
                            nn.Linear(num_nonfinal_states//2, 2),  # Output layer
                        )

                    # Initialize weights in the range [0, 0.1]
                    for layer in self.model:
                        if isinstance(layer, nn.Linear):
                            nn.init.uniform_(layer.weight, 0, init_max)
                            nn.init.uniform_(layer.bias, 0, init_max)
                
                def forward(self, x):
                    return self.model(x)

            # Helper function to get all Q-values (similar to get_all_q_values in original code)
            def get_all_q_values(agent, num_nonfinal_states, num_actions=2):
                all_q_values = np.zeros((num_nonfinal_states, num_actions))
                for state in range(num_nonfinal_states):
                    for action in range(num_actions):
                        # Prepare input for the agent (state, action pair)
                        one_hot_state = np.zeros(num_nonfinal_states + 1)
                        one_hot_state[state] = 1
                        one_hot_state[-1] = 1
                        input_tensor = torch.tensor(np.expand_dims(one_hot_state, axis=0), dtype=torch.float32)
                        # Get the Q-value prediction
                        all_q_values[state, action] = agent(input_tensor).detach().numpy()[0][action]  # Convert tensor to scalar

                return all_q_values

            results = {method:[] for method in methods}
            all_sampled_idxes = {i:[] for i in methods}
            q_val_traces = {i:[] for i in methods}
            sampling_proba_traces = {i:[] for i in methods}
            td_error_traces = {i:[] for i in methods}
            deltas = {i:[] for i in methods}


            assert (episodes == sorted(episodes))

            for method in methods:

                start = time.process_time()

                for seed in tqdm(range(iterations)):

                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    counter = 0
                    sampled_idxes = []
                    q_val_trace = []
                    sampling_proba_trace = []
                    td_error_trace = []
                    

                    td_errors = np.ones_like(states, dtype=float)

                    # Initialize the agent
                    if approximator == "TQL":
                        learned_q_values = np.random.uniform(0, init_max, size=(num_nonfinal_states, 2))
                    if approximator in ["DQN", "LFA"]:
                        agent = QLearningAgent()
                        optimizer = optim.SGD(agent.parameters(), lr=lr)
                        loss_fn = nn.MSELoss()
                        learned_q_values = get_all_q_values(agent, num_nonfinal_states)
                    
                    for _ in range(num_training_steps):

                        if approximator in ["DQN", "LFA"]:
                            idx, state, action, IS_weight, sampling_probas = get_sampling_idx(method, td_errors, episodes, alpha, use_IS)

                            # Prepare the input and target for training
                            one_hot_state = np.zeros(num_nonfinal_states + 1)
                            one_hot_state[state] = 1
                            one_hot_state[-1] = 1
                            x_tensor = torch.tensor(np.expand_dims(one_hot_state, axis=0), dtype=torch.float32)
                            
                            if dones[idx]:  # Check if episode is done
                                target = rewards[idx]
                            else:
                                target = rewards[idx] + max(learned_q_values[next_states[idx]]) * discount_factor
                            
                            optimizer.zero_grad()
                            predicted_q_value = agent(x_tensor).squeeze()  # Squeeze to get rid of single dimension
                            target_tensor = predicted_q_value.clone()
                            target_tensor[action] = target
                            loss = loss_fn(predicted_q_value, target_tensor) * torch.tensor(IS_weight)
                            loss.backward()
                            optimizer.step()

                            td_error = abs(target_tensor - predicted_q_value).sum().item() + 1e-6
                            learned_q_values = get_all_q_values(agent, num_nonfinal_states)

                        elif approximator == "TQL":

                            idx, state, action, IS_weight, sampling_probas = get_sampling_idx(method, td_errors, episodes, alpha, use_IS)
                            predicted_q_value = learned_q_values[state, action]

                            if dones[idx]:  # Check if episode is done
                                target = rewards[idx]
                            else:
                                target = rewards[idx] + max(learned_q_values[next_states[idx]]) * discount_factor
                            learned_q_values[state, action] += lr * (target - predicted_q_value)

                            td_error = abs(target - predicted_q_value) + 1e-6

                        td_errors[idx] = td_error
                        counter += 1
                    
                    delta = np.mean(target_q_values - learned_q_values)
                    deltas[method].append(delta)

                print(f"Max. episode length = {episode_length}; time={round(time.process_time() - start, 2)}")
                print(f"Delta: {np.mean(deltas[method])}")

                if method == "ReaPER":
                    if ("ReaPER" in methods) and ("UNI" in methods):
                        print(f'ReaPER vs UNI Edge: {(np.mean(deltas["UNI"]) - np.mean(deltas["ReaPER"])) / np.mean(deltas["UNI"])}')
                    if ("ReaPER" in methods) and ("PER" in methods):
                        print(f'ReaPER vs PER Edge: {(np.mean(deltas["PER"]) - np.mean(deltas["ReaPER"])) / np.mean(deltas["PER"])}')

                        
            print("----------------")