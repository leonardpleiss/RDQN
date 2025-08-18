import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
from tqdm import tqdm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


trial_lengths = np.arange(1, 10) * 10
num_actions = 2
final_reward = 1
deterministic = True
iterations = 10
max_init = 1.

Q_value_inits = ["High target reliability", "Medium target reliability", "Low target reliability"]# "High target reliability"

strategies = [
    "Oracle",
    "Uniform",
    "PER-g",
    "ReaPER-g",
    ]

colors = {
    "PER-g": "black",
    "Uniform": "lightgrey",
    "ReaPER-g": "red",
    "Oracle": "green"
}

linestyles = {
    "PER-g": "-",
    "Uniform": "-",
    "ReaPER-g": ":",
    "Oracle": "-"
}

linewidths = {
    "PER-g": 2,
    "Uniform": 2,
    "ReaPER-g": 2,
    "Oracle": 2,
}

alphas = {
    "PER-g": 1.,
    "Uniform": 1.,
    "ReaPER-g": 1.,
    "Oracle": 1.,
}

def adjust_Q(idx, action_Qs, target_Qs):
    action_Qs[idx] = target_Qs[idx]
    return action_Qs

def get_transition_idx(action_Qs, target_Qs, strategy, final_reward):

    TDs = abs(action_Qs - target_Qs)

    num_transitions = len(action_Qs)

    if strategy == "Uniform":
        return np.random.choice(np.arange(num_transitions))

    elif strategy == "PER-g":

        if deterministic:
            return np.argmax(TDs)
        else:
            sampling_probas = TDs/TDs.sum()
            return np.random.choice(np.arange(num_transitions), p=sampling_probas)
    
    elif strategy == "ReaPER-g":
        rel_weight = TDs.cumsum() / TDs.sum()
        sampling_weight = TDs * rel_weight
        
        if deterministic:
            return np.argmax(sampling_weight)
        else:
            sampling_probas = sampling_weight / sampling_weight.sum()
            return np.random.choice(np.arange(num_transitions), p=sampling_probas)
        
    elif strategy == "Oracle":
        idx = np.where(action_Qs != final_reward)[0][-1]
        return idx
    
fig, axs = plt.subplots(ncols=len(Q_value_inits), nrows=1, figsize=(8.27, 2.4))
for plot_idx, Q_value_init in enumerate(Q_value_inits):

    results = {}

    for strategy in tqdm(strategies):

        results[strategy] = {}

        for trial_length in trial_lengths: 
            
            results[strategy][trial_length] = []
            
            for iteration in range(iterations):

                counter = 0
                
                np.random.seed(iteration)
                
                actions = np.ones(shape=(trial_length,), dtype=int)

                if Q_value_init == "Low target reliability":
                    Q_values = np.zeros(shape=(trial_length, 2), dtype=float)
                    action_Qs = np.array([i%2==0 for i in range(trial_length)]).astype(int)
                    Q_values[np.arange(len(actions)), actions] = action_Qs
                    
                    target_Qs = np.max(Q_values, axis=1)[1:]
                    target_Qs = np.append(target_Qs, final_reward)
                    
                if Q_value_init == "Medium target reliability":
                    Q_values = np.zeros(shape=(trial_length, 2), dtype=float)
                    action_Qs = np.array([i%4==0 for i in range(trial_length)]).astype(int)
                    Q_values[np.arange(len(actions)), actions] = action_Qs
                    
                    target_Qs = np.max(Q_values, axis=1)[1:]
                    target_Qs = np.append(target_Qs, final_reward)

                if Q_value_init == "High target reliability":
                    Q_values = np.zeros(shape=(trial_length, 2), dtype=float)
                    action_Qs = np.zeros(trial_length)
                    Q_values[np.arange(len(actions)), actions] = action_Qs
                    
                    target_Qs = np.max(Q_values, axis=1)[1:]
                    target_Qs = np.append(target_Qs, final_reward)
                
                while (action_Qs != final_reward).any():
                    counter += 1
                    
                    transition_idx = get_transition_idx(action_Qs, target_Qs, strategy=strategy, final_reward=final_reward)
                    action_Qs = adjust_Q(transition_idx, action_Qs, target_Qs)

                    # Update targets
                    Q_values[np.arange(len(actions)), actions] = action_Qs
                    target_Qs = np.max(Q_values, axis=1)[1:]
                    target_Qs = np.append(target_Qs, final_reward)

                results[strategy][trial_length].append(counter)
                # print(counter)

    step_avg, step_min, step_max = {i:[] for i in strategies}, {i:[] for i in strategies}, {i:[] for i in strategies}
    for strategy in strategies:
        
        for trial_length in trial_lengths:
            steps = results[strategy][trial_length]
            step_avg[strategy].append(np.mean(steps))
            step_min[strategy].append(np.min(steps))
            step_max[strategy].append(np.max(steps))

    for strategy in [i for i in strategies if i != "Oracle"]:

        color = colors[strategy]
        linewidth = linewidths[strategy]
        alpha = alphas[strategy]
        linestyle = linestyles[strategy]
        markersize = 3

        x = np.array(trial_lengths)
        y = np.array(step_avg[strategy]) - np.array(step_avg["Oracle"])

        axs[plot_idx].plot(x, y, color=color, label=strategy, linestyle=linestyle, linewidth=linewidth, alpha=alpha) #, marker="o", markersize=markersize)
        
        if not deterministic:
            axs[plot_idx].fill_between(trial_lengths, step_min, step_max, color=color, alpha=.3)

    if plot_idx == 0:
        axs[plot_idx].set_ylabel("# required updates\n- # min. required updates")#, fontsize=12)
    axs[plot_idx].set_xlabel("Episode length")
    axs[plot_idx].grid()
    axs[plot_idx].set_title(Q_value_init)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(strategies), frameon=True)

fig.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig("result_analysis/saved_figures/single_trajectory_simul.png", dpi=500)
plt.show()