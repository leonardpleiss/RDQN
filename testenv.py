import numpy as np
import sys
import pickle
import os
import matplotlib.pyplot as plt
import torch as th

current_q_values = th.zeros(3)

target_q_values_ddqn = th.randn(3) * 100# th.tensor([-10, -3, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2.1])
target_q_values_online = th.randn(3) * 100#th.tensor([-9, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2.1, 2])

ddqn_err = target_q_values_ddqn - current_q_values
online_err = target_q_values_online - current_q_values
new_online_err = target_q_values_online - (current_q_values + ddqn_err)

online_error_change =  th.abs(online_err) / (th.abs(online_err) + th.abs(new_online_err))
online_more_extreme = ( (ddqn_err * online_err) > 0) & (th.abs(online_err) > th.abs(ddqn_err))

weight = th.where(online_more_extreme, 1, online_error_change)
# print(f"{target_q_values_ddqn=}")
# print(f"{target_q_values_online=}")
print(f"{ddqn_err=}")
print(f"{online_err=}")
print(f"{new_online_err=}")
print(f"{online_error_change=}")
print(f"{weight=}")
# print(f"{online_err_increase_raw=}") 

# Learnings: DDQN-Error braucht 1e-8













































sys.exit()
y = th.tensor([[0.0673, 0.0721, 0.0779],
        [0.0470, 0.0113, 0.0233]])
x = th.tensor([[ 0.0108,  0.0269, -0.0109],
        [ 0.0007,  0.0324, -0.0185]])

print(th.sum(th.abs(x) + th.abs(y), dim=1))
# tensor([[0.1906],
#         [0.1092]])
# tensor([[0.1330],
#         [0.0667]])
sys.exit()
iters = 50

n_action_lst = [2, 4, 8, 16, 32, 64]
trajectory_length_lst = [10, 100, 1000]

fig, axs = plt.subplots(nrows=len(n_action_lst), ncols=len(trajectory_length_lst))



for row_idx, n_actions in enumerate(n_action_lst):

    for col_idx, trajectory_length in enumerate(trajectory_length_lst):

        inits = []
        for _ in range(iters):

            init = np.random.normal(0, .1, (trajectory_length, n_actions))
            num_updates = trajectory_length * 1

            for _ in range(num_updates):

                transition_idx = np.random.randint(trajectory_length)
                action_idx = np.argmax(init[transition_idx])

                if transition_idx == (trajectory_length-1):
                    init[transition_idx, action_idx] = 0
                else:
                    init[transition_idx, action_idx] = np.max(init[transition_idx + 1]) * .99

            inits.append(np.max(init, axis=1))
        
        avg_inits = np.array(inits).mean(axis=0)
        print(row_idx, col_idx)
        axs[row_idx, col_idx].set_title(f"{n_actions} actions across {trajectory_length} timesteps")

        axs[row_idx, col_idx].plot(np.arange(trajectory_length), avg_inits)

plt.tight_layout()
plt.show()
        




sys.exit()
# max_subsequent_td = 5000

# tds = [1, 1, 1, 1, 1, 1, 1, 1, 1]
# subsequent_tds = np.array([0, .5, 1, 5, 10, 50, 100, 500, 1000])


# simple_weights = (np.max(subsequent_tds) - subsequent_tds) / subsequent_tds.sum()

# print(simple_weights)

# sys.exit()


# max_subsequent_tds = max(max_subsequent_td, np.max(subsequent_tds))
# weights = np.exp(-subsequent_tds / max_subsequent_tds)
# weights /= weights.mean()

# print(f"{weights=}")
# sys.exit()

# logged_subsequent_tds = np.log1p(subsequent_tds)

# # plt.hist(subsequent_tds)
# plt.hist(logged_subsequent_tds)
# plt.show()
# z_l_subsequent_tds = (logged_subsequent_tds - logged_subsequent_tds.mean()) / logged_subsequent_tds.std()

# c_z_l_subsequent_tds = np.clip(z_l_subsequent_tds, a_min=-2, a_max=2)

# weights = 1 + (c_z_l_subsequent_tds / 4)

# print(f"{logged_subsequent_tds=}")
# print(f"{z_l_subsequent_tds=}")
# print(f"{c_z_l_subsequent_tds=}")
# print(f"{weights=}")


# sys.exit()

# store_path = "../weight_history_CP.pkl"
# print(os.listdir())

# with open(store_path, 'rb') as f:
#     weight_history = pickle.load(f)


# num_splits = 4

# weight_history = np.array(weight_history).reshape(-1, 64)

# parts = np.array_split(weight_history, num_splits, axis=0)


# fig, axs = plt.subplots(nrows=num_splits, ncols=1)

# for idx, part in enumerate(parts):
#     flat_part = part.reshape(-1)
#     axs[idx].hist(flat_part)
# plt.show()


# sys.exit()
# max_subs = 1200

# subs = np.array([0, .5, 1, 5, 10, 50, 1000])
# # subs = np.array([0, 1, 2, 3])

# subs = np.array([0, 10, 200, 500, 1000])

# # z_subs = subs / subs.std()


# subs_inv = 1 / (1 + subs / max_subs)

# weights = subs_inv / subs_inv.mean()

# print(subs)
# print(subs_inv)
# # print(reg_subs_inv)
# print(weights)


# sys.exit()

# LL_DDQN_BL = [13, 28, 19, 45, 23, 33, 24, 42, 19, 20]
# LL_DQN_BL = [47, 15, 25, 40, 15, 37, 39, 22, 23, 24]
LL_DDQN_BL = [26, 14, 67, 40, 70, 14, 33, 36, 19, 36]
LL_DQN_BL = [22, 33, 37, 20, 45, 20, 16, 18, 21, 24]

CP_DDQN_BL = [4.5, 23, 27, 29.5, 10.5, 19.5, 16.5, 29, 23.5, 13.5]
CP_DQN_BL = [19.5, 26, 32.5, 34, 2, 30.5, 17, 10.5, 26, 23.5]

AC_DDQN_BL = [16, 23, 17, 23, 20, 12, 20, 12, 15, 10]
AC_DQN_BL = [23, 30, 22, 10, 13, 20, 24, 30, 11, 23]
# ======================================================================================= #

CP_CLIP = [11, 27.5, 5, 22.5, 38, 31, 7.5, 18.5, 6.5, 19.5]
CP_SCALE = [30.5, 20.5, 34, 30, 2, 6.5, 14, 9.5, 26, 20.5]
CP_IS_1 = [7, 26, 22, 20.5, 7.5, 25, 22.5, 24, 14, 6.5]
CP_IS_5 = [24, 29, 29.5, 31, 28.5, 17, 29.5, 6, 5.5, 19.5]
CP_LS_1 = [4.5, 25.5, 22.5, 27, 25.5, 27, 31, 37, 9, 22.5]
CP_IS_ = [8, 2, 16.5, 17, 24.5, 17.5, 25.5, 23.5, 23.5, 39.5, 7.5]
CP_IS_2 = [27.5, 10, 9, 24, 14, 6, 25, 27.5, 6.5, 30.5]
CP_IS_3 = [7, 19, 24, 33, 24, 24, 8, 17.5, 11.5, 12]
CP_IS_3_SQU = [34, 13, 28, 9, 28, 27.5, 15.5, 18, 5, 15]
CP_IS_3_ND = [30, 7.5, 18, 25.5, 2, 19.5, 19, 12.5, 8, 4.5]
CP_IV_BLEND = [5.5, 15, 7.5, 12, 16, 16, 12.5, 8.5, 21, 12.5]
CP_IV_BLEND_OL = [7, 4, 4.5, 4.5, 8.5, 5, 3.4, 4.5, 6.5, 5.5]
CP_IV_BLEND_ND = [9.5, 13.5, 15.5, 7, 11, 15, 17, 9, 8.5, 18]

AC_CLIP = [26, 23, 28, 9, 11, 23, 27, 15, 11, 25]
AC_SCALE = [28, 9, 14, 10, 21, 14, 10, 23, 12, 11]
AC_IS_5 = [12, 10, 12, 27, 19, 17, 33, 10, 30, 30]
AC_IS_2 = [19, 17, 9, 17, 22, 15, 20, 10, 10, 12]
AC_IS_3 = [10, 18, 19, 26, 15, 6, 21, 10, 19, 15]
AC_IS_3_ND = [26, 31, 10, 26, 19, 17, 26, 13, 14, 19]
AC_IV_BLEND = [6, 13, 6, 15, 13, 13, 8, 19, 15, 17]
AC_IV_BLEND_OL = [7, 4, 5, 5, 3, 4, 10, 2, 5, 4]
AC_IV_BLEND_ND = [7, 3, 5, 13, 4, 5, 15, 6, 4, 7]

LL_CLIP = [38, 11, 50, 34, 18, 8, 12, 56, 16, 21]
LL_SCALE = [31, 66, 38, 42, 7, 26, 21, 21, 17, 24]
LL_IS_5 = [41, 39, 13, 18, 55, 19, 14, 20, 27, 50]
LL_IS_3 = [15, 35, 13, 47, 32, 16, 12, 33, 16, 21]
LL_IV_BLEND = [26, 31, 14, 17, 36, 27, 15, 26, 15, 23]
LL_IV_BLEND_OL = [16, 14, 22, 44, 31, 45, 10, 17, 30, 19]
LL_IV_BLEND_ND = [21, 40, 29, 24, 25, 19, 16, 14, 20, 23]

# print( np.mean(np.array(LL_DDQN_BL) / 100.000) )
# print( np.mean(np.array(AC_DDQN_BL) / 50.000) )
# print( np.mean(np.array(CP_DDQN_BL) / 50.000) )
# sys.exit()

print("LunarLander")
print(np.mean(LL_DQN_BL))
print(np.mean(LL_DDQN_BL))
print(np.mean(LL_CLIP))
print(np.mean(LL_SCALE))
print(np.mean(LL_IS_3))
print(np.mean(LL_IV_BLEND))
print(np.mean(LL_IV_BLEND_OL))
print(np.mean(LL_IV_BLEND_ND))

print()

print("Acrobot")
print(np.mean(AC_DQN_BL))
print(np.mean(AC_DDQN_BL))
print(np.mean(AC_CLIP))
print(np.mean(AC_SCALE))
print(np.mean(AC_IS_2))
print(np.mean(AC_IS_3))
print(np.mean(AC_IV_BLEND))
print(np.mean(AC_IV_BLEND_OL))
print(np.mean(AC_IV_BLEND_ND))

print()

print("CartPole")
print(np.mean(CP_DQN_BL))
print(np.mean(CP_DDQN_BL))
print(np.mean(CP_CLIP))
print(np.mean(CP_SCALE))
print(np.mean(CP_LS_1))
print(np.mean(CP_IS_))
print(np.mean(CP_IS_3))
print(np.mean(CP_IV_BLEND))
print(np.mean(CP_IV_BLEND_OL))
print(np.mean(CP_IV_BLEND_ND))



def percent_lower(current, reference):
    """
    Returns how much lower `current` is than `reference` as a percentage.
    
    Parameters:
    - current: the score you want to compare
    - reference: the score to compare against
    
    Example:
    >>> percent_lower(80, 100)
    20.0
    """
    if reference == 0:
        raise ValueError("Reference score cannot be zero.")
    return ((reference - current) / reference) * 100

envs = ["CartPole", "Acrobot", "LunarLander"]
DQN_steps = [22.15, 20.6, 28.7]
DDQN_steps = [19.65, 16.8, 26.6]
CUSTOM_steps = [12.65, 12.65, 23.0]
CUSTOM_steps = [18.0, 15.9, 24.0]

for idx, custom in enumerate(CUSTOM_steps):

    print(envs[idx])
    percent_lower_dqn = percent_lower(custom, DQN_steps[idx])
    percent_lower_ddqn = percent_lower(custom, DDQN_steps[idx])
    print(f"DQN: {percent_lower_dqn}")
    print(f"DDQN: {percent_lower_ddqn}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

env_dict = {
    "LunarLander": [LL_DQN_BL, LL_DDQN_BL, LL_IV_BLEND],
    "CartPole": [CP_DQN_BL, CP_DDQN_BL, CP_IV_BLEND],
    "Acrobot": [AC_DQN_BL, AC_DDQN_BL, AC_IV_BLEND],
}

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(12,7))

for idx, env in enumerate(env_dict):

    labels = ['DQN', 'DDQN', 'ReaDQN']
    colors = ['black', 'grey', 'red']
    alpha = 0.4

    # Create the violin plot
    parts = axs[idx].violinplot(env_dict[env], showmeans=False, showmedians=False, showextrema=False)

    # Set violin body colors with alpha
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(alpha)

    # Plot horizontal mean lines in solid color
    for i, array in enumerate(env_dict[env]):
        mean = np.mean(array)
        axs[idx].hlines(y=mean, xmin=i + 0.75, xmax=i + 1.25, color=colors[i], linewidth=2.5, label=f'{labels[i]} Mean')

    # Formatting
    axs[idx].set_xticks([1, 2, 3])
    axs[idx].set_xticklabels(labels)
    axs[idx].set_ylabel('k steps until completion')
    axs[idx].set_title(f'{env}: Steps until completion')
    axs[idx].grid(True, axis='y')
    # axs[idx].grid(True, which='', linestyle='--', linewidth=0.5) 
    
plt.tight_layout()
# plt.show()








# CP_UNI = [
#     20, 26, 27, 32, 28, 24, 32, 24, 8, 9, 26, 26, 33, 34, 2, 31, 17, 11, 26, 24
# ]

# CP_R_UNI_10per = [
#     4, 30, 30, 30, 23, 24, 17, 21, 28, 42, 31, 27, 29, 9, 7, 28, 34, 13, 31, 23
# ]


# AC_UNI = [
#     23, 11, 29, 20, 27, 24, 22, 12, 28, 11, 29, 30, 22, 10, 13, 20, 24, 30, 11, 23,
# ]

# AC_R_UNI_10per = [
#     16, 8, 21, 7, 22, 17, 10, 11, 25, 44, 11, 14, 7, 17, 28, 13, 17, 25, 25, 16
# ]


# LL_UNI = [
#     31, 29, 24, 27, 43, 38, 17, 21, 
# ]

# LL_R_UNI_10per = [
#     30, 37, 53, 68, 30, 18, 19, 56, 
# ]

# print(np.mean(CP_UNI))
# print(np.mean(CP_R_UNI_10per))

# print(np.mean(AC_UNI))
# print(np.mean(AC_R_UNI_10per))

# print(np.mean(LL_UNI))
# print(np.mean(LL_R_UNI_10per))