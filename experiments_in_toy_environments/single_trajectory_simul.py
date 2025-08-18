import matplotlib.pyplot as plt
import numpy as np
import copy
import sys

# --------------------------------- #

# episode_length = 100
deterministic = False
iters = 30
init_max = .5
strategies = [
    # "UNI",
    "PER",
    "ReaPER",
    ]

max_lenfac = 14

# --------------------------------- #
 
colors = {
    "PER":"black",
    "UNI":"grey",
    "ReaPER":"red",
}

np.random.seed(0)

def adjust_Q(idx, estimated_Q):
    estimated_Q[idx] = estimated_Q[idx+1]
    return estimated_Q

def get_transition_idx(estimated_Q, strategy):
    num_transitions = len(estimated_Q) - 1

    if strategy == "UNI":
        return np.random.choice(np.arange(num_transitions))

    elif strategy == "PER":
        transition_TDs = abs(estimated_Q[:-1] - estimated_Q[1:])
        sampling_probas = transition_TDs/transition_TDs.sum()

        if deterministic:
            return np.argmax(sampling_probas)
        else:
            return np.random.choice(np.arange(num_transitions), p=sampling_probas)
    
    elif strategy == "ReaPER":
        transition_TDs = abs(estimated_Q[:-1] - estimated_Q[1:])
        rel_weight = transition_TDs.cumsum() / transition_TDs.sum()
        sampling_weight = transition_TDs * rel_weight
        sampling_probas = sampling_weight / sampling_weight.sum()
    
        if deterministic:
            return np.argmax(sampling_probas)
        else:
            return np.random.choice(np.arange(num_transitions), p=sampling_probas)

strategy_mean_dict = {strategy:[] for strategy in strategies}


episode_lengths =  (2 ** np.arange(1, max_lenfac))
print(2**10)
for episode_length in episode_lengths:

    strategy_stepsize_dict = {strategy:[] for strategy in strategies}


    real_Q = np.ones(episode_length, dtype=np.int64)
    estimated_Q = np.random.uniform(0, init_max, size=(episode_length,))
    estimated_Q[-1] = real_Q[-1]

    for strategy in strategy_stepsize_dict:
        for _ in range(iters):
            counter = 0
            Q_values_to_transform = copy.deepcopy(estimated_Q)

            while (Q_values_to_transform != real_Q).any():
                counter += 1
                transition_idx = get_transition_idx(Q_values_to_transform, strategy=strategy)
                Q_values_to_transform = adjust_Q(transition_idx, Q_values_to_transform)
            
            strategy_stepsize_dict[strategy].append(counter)

    print(f"{episode_length=}")
    for strategy in strategy_stepsize_dict:
        mean = np.mean(strategy_stepsize_dict[strategy])
        print(f"{strategy}: {mean.round(2)}")
        strategy_mean_dict[strategy].append(mean)

    if ("PER" in strategy_stepsize_dict) & ("ReaPER" in strategy_stepsize_dict):
        print(f'ReaPER vs. PER Edge: {np.round((np.mean(strategy_stepsize_dict["PER"]) - np.mean(strategy_stepsize_dict["ReaPER"])) / np.mean(strategy_stepsize_dict["PER"]), 2)}')

    if ("UNI" in strategy_stepsize_dict) & ("ReaPER" in strategy_stepsize_dict):
        print(f'ReaPER vs. Uni Edge: {np.round((np.mean(strategy_stepsize_dict["UNI"]) - np.mean(strategy_stepsize_dict["ReaPER"])) / np.mean(strategy_stepsize_dict["UNI"]), 2)}')

    print("---")


fig, axs = plt.subplots()
for strategy in strategy_mean_dict:
    axs.plot(episode_lengths, strategy_mean_dict[strategy], label=strategy, color=colors[strategy])

axs.legend()
axs.set_ylabel("Number of steps in episode")
axs.set_xlabel("Average number of update steps required")
axs.set_xscale('log')
# axs.set_yscale('log')
axs.grid()
plt.show()