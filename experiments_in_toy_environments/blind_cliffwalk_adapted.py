import numpy as np
import copy
import sys
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

####################################################### PARAMETERS #######################################################

# Experiment parameters
methods = ["per", "reaper"] # Available are "uni", "per", "reaper"
approximators = ["TQL", "LFA", "DQN"]#, "DQN"] # "DQN" or "LFA" or "TQL"
max_episode_length = 3                           
cutoff_timesteps = 1_500_000
iterations = 10

# Run parameters
init_max = 1.
alpha = 1.
alpha2 = 1.
use_IS = False
tol = 1e-3
seed = 0
lr = .25
stochastic = False

##########################################################################################################################

for stochastic in [True, False]:
    for approximator in approximators:
        result_dict = {
            "episode_length": [],
            "num_transitions": [],
            "uni_mean": [],
            "uni_sd": [],
            "uni_min": [],
            "uni_max": [],
            "per_mean": [],
            "per_sd": [],
            "per_min": [],
            "per_max": [],
            "reaper_mean": [],
            "reaper_sd": [],
            "reaper_min": [],
            "reaper_max": [],
        }

        for episode_length in np.arange(2, max_episode_length+1, 1):
            
            discount_factor = 1. - (1/episode_length)
            print(f"{discount_factor=}")

            np.random.seed(seed)

            states, actions, episodes, rewards, dones = [], [], [], [], []

            episode = 0
            # correct_path = np.ones((episode_length), dtype=int)
            correct_path = np.random.randint(2, size=(episode_length))

            # Ensure that there is no trivial solution
            correct_path[0] = 1
            correct_path[-1] = 0

            target_q_values = np.zeros((episode_length, 2))
            target_q_values[np.arange(episode_length), correct_path] = 1
            target_q_values[np.arange(episode_length), correct_path] = (
                target_q_values * discount_factor ** np.expand_dims(np.arange(episode_length-1, -1, -1), axis=1)
                )[np.arange(episode_length), correct_path] # Apply discount

            # Fill buffer
            action_plans = np.array(list(itertools.product([0, 1], repeat=episode_length)))

            for action_plan in action_plans:
                state = 0
                episode_done = False

                while not episode_done:

                    action = action_plan[state]
                    action_is_right = action == correct_path[state]

                    states.append(state)
                    actions.append(action)
                    episodes.append(episode)

                    reward = 0
                    if action_is_right:
                        
                        reward = 1

                        if (state == episode_length-1):
                            episode += 1
                            episode_done = True
                            path_found = True
                        else:
                            state += 1

                    else:
                        episode += 1
                        state = 0
                        episode_done = True

                    rewards.append(reward)
                    dones.append(episode_done)

            dones = np.array(dones, dtype=int)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            print(f"Number of transitions: {len(states)}")
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

                if method == "uni":
                    sampling_probas = np.ones_like(num_transitions) / len(num_transitions)
                    idx = np.random.choice(num_transitions)
                    
                elif method == "per":
                    sampling_weights = td_errors ** alpha
                    sampling_probas = sampling_weights / sampling_weights.sum()

                    if stochastic:
                        idx = np.random.choice(num_transitions, p=sampling_probas)
                    else:
                        idx = np.argmax(sampling_probas)

                elif method == "reaper":
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
                            nn.Linear(episode_length+1, 2)  # Output layer
                        )

                    if approximator == "DQN":
                        self.model = nn.Sequential(
                            nn.Linear(episode_length+1, episode_length//2),
                            nn.ReLU(),
                            nn.Linear(episode_length//2, 2),  # Output layer
                        )

                    # Initialize weights in the range [0, 0.1]
                    for layer in self.model:
                        if isinstance(layer, nn.Linear):
                            nn.init.uniform_(layer.weight, 0, init_max)
                            nn.init.uniform_(layer.bias, 0, init_max)
                
                def forward(self, x):
                    return self.model(x)

            # Helper function to get all Q-values (similar to get_all_q_values in original code)
            def get_all_q_values(agent, episode_length, num_actions=2):
                all_q_values = np.zeros((episode_length, num_actions))
                for state in range(episode_length):
                    for action in range(num_actions):
                        # Prepare input for the agent (state, action pair)
                        one_hot_state = np.zeros(episode_length + 1)
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

            assert (episodes == sorted(episodes))

            for method in methods:

                start = time.process_time()

                for seed in range(iterations):

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
                        learned_q_values = np.random.uniform(0, init_max, size=(episode_length, 2))
                    if approximator in ["DQN", "LFA"]:
                        agent = QLearningAgent()
                        optimizer = optim.SGD(agent.parameters(), lr=lr)
                        loss_fn = nn.MSELoss()
                        learned_q_values = get_all_q_values(agent, episode_length)
                    
                    # Loop until target Q-values and learned Q-values match
                    while not np.isclose(np.mean((target_q_values - learned_q_values)**2), 0, atol=tol).all():
                    # while not np.isclose(target_q_values, learned_q_values, atol=tol).all():
                    # while (np.argmax(learned_q_values, axis=1) != correct_path).any(): 

                        # print(f"{learned_q_values=}")
                        # print(f"{target_q_values=}")

                        if approximator in ["DQN", "LFA"]:
                            idx, state, action, IS_weight, sampling_probas = get_sampling_idx(method, td_errors, episodes, alpha, use_IS)

                            # Prepare the input and target for training
                            one_hot_state = np.zeros(episode_length + 1)
                            one_hot_state[state] = 1
                            one_hot_state[-1] = 1
                            x_tensor = torch.tensor(np.expand_dims(one_hot_state, axis=0), dtype=torch.float32)
                            
                            if dones[idx]:  # Check if episode is done
                                target = rewards[idx]
                            else:
                                target = rewards[idx] + max(learned_q_values[state+1]) * discount_factor
                            
                            optimizer.zero_grad()
                            predicted_q_value = agent(x_tensor).squeeze()  # Squeeze to get rid of single dimension
                            target_tensor = predicted_q_value.clone()
                            target_tensor[action] = target
                            loss = loss_fn(predicted_q_value, target_tensor) * torch.tensor(IS_weight)
                            loss.backward()
                            optimizer.step()

                            td_error = abs(target_tensor - predicted_q_value).sum().item() + 1e-6
                            learned_q_values = get_all_q_values(agent, episode_length)

                        elif approximator == "TQL":

                            idx, state, action, IS_weight, sampling_probas = get_sampling_idx(method, td_errors, episodes, alpha, use_IS)
                            predicted_q_value = learned_q_values[state, action]
                            
                            # print("Vor Update")
                            # print(f"{idx=}")
                            # print(f"{states=}")
                            # print(f"{actions=}")
                            # print(f"{correct_path=}")
                            # print(f"{state, action=}")
                            # print(f"{td_errors.round(3)=}")
                            # print(f"{sampling_probas.round(3)=}")
                            # print("---")
                        
                            if dones[idx]:  # Check if episode is done
                                target = rewards[idx]
                            else:
                                target = rewards[idx] + max(learned_q_values[state+1]) * discount_factor
                            learned_q_values[state, action] += lr * (target - predicted_q_value)

                            td_error = abs(target - predicted_q_value) + 1e-6

                        td_errors[idx] = td_error
                        counter += 1

                        # print("Nach Update")
                        # print(f"{td_errors.round(3)=}")
                        # print(f"{sampling_probas.round(3)=}")
                        # print("-----------")

                        if counter>cutoff_timesteps:
                            break
                    
                    results[method].append(counter)
                    all_sampled_idxes[method] = sampled_idxes
                    q_val_traces[method] = np.array(q_val_trace)
                    sampling_proba_traces[method] = np.array(sampling_proba_trace)
                    td_error_traces[method] = np.array(td_error_trace)

                result_dict[f"{method}_mean"].append(np.mean(results[method]))
                result_dict[f"{method}_sd"].append(np.std(results[method]))
                result_dict[f"{method}_min"].append(np.min(results[method]))
                result_dict[f"{method}_max"].append(np.max(results[method]))

                print(f"Max. episode length = {episode_length}; time={round(time.process_time() - start, 2)}")
                print(f"{method}: M = {np.round(np.mean(results[method]),2)} (SD = {np.round(np.std(results[method]),2)}, Min = {np.round(np.min(results[method]),2)}, Max = {np.round(np.max(results[method]),2)})")

                if method == "reaper":
                    if ("reaper" in methods) and ("per" in methods):
                        reaper_per_edge = (np.mean(results["per"]) - np.mean(results["reaper"])) / np.mean(results["per"]) 
                        prop_of_runs_better = (np.array(results["per"]) > np.array(results["reaper"])).sum() / iterations
                        print(f"Prop of runs with Reaper outperformance: {np.round(prop_of_runs_better, 1)}")
                        print(f"Reaper vs PER Edge: {np.round(reaper_per_edge, 2)}")

                    if ("reaper" in methods) and ("uni" in methods):
                        reaper_uni_edge = (np.mean(results["uni"]) - np.mean(results["reaper"])) / np.mean(results["per"]) 
                        print(f"Reaper vs UNI Edge: {np.round(reaper_uni_edge, 2)}")
            print("----------------")

        # Define colors for each method
        colors = {
            "uni": "blue",
            "reaper": "red",
            "per": "black"
        }

        fig, ax = plt.subplots()
        # Plot each method with its mean line and shaded area for min/max range
        for method in methods:
            x = result_dict["num_transitions"]
            y_mean = result_dict[f"{method}_mean"]
            y_min = result_dict[f"{method}_min"]
            y_max = result_dict[f"{method}_max"]

            ax.plot(x, y_mean, label=method, color=colors[method])  # Line plot
            ax.fill_between(x, y_min, y_max, color=colors[method], alpha=0.2)  # Shaded area

        # Labels and legend
        ax.set_title(f"BlindCliffwalk of varying size using {approximator}; stochastic={stochastic}")
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_ylabel("Number of transitions")
        ax.set_xlabel("Number of updates required")
        ax.legend()

plt.show()