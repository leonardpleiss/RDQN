import numpy as np
import sys


# reliability = np.arange(101)/100
# _alpha2 = 1.5
# reliability = _alpha2 ** (2 * reliability - 1)
# print(reliability)

# # target = [5.5, 8.5, 5.5, 7, 14, 34, 31, 27, 27, 32, 24, 24, 33, 31, 28, 8.5, 7.5, 32, 5.5, 31]

LL_DDQN_BL = [13, 28, 19, 45, 23, 33, 24, 42, 19, 20]
LL_DQN_BL = [47, 15, 25, 40, 15, 37, 39, 22, 23, 24]
LL_RELCLIP = [33, 45, 12, 30, 12, 17, 11, 33, 25, 22]
LL_RELCLIP_LOWERMEANRETURN_UPPERMAXRETURN = [20, 67, 32, 13, 34, 22, 37, 26, 19, 51]

LL_RELCLIP_LOWER_HALFMAXTD_UPPER_MAXTD = [26, 28, 46, 51, 26, 53, 23, 19, 16, 19]
LL_RELCLIP_MAXTDTIMESPROPREL = [33, 33, 19, 50, 20, 12, 23, 34, 40, 12] # clip_target_historically

LL_RELCLIP_MAXTD = [39, 30, 21, 14, 71, 13, 13, 32, 13, 37]
LL_RELCLIP_SCALEDBOOTSTRAP = [21, 17, 12, 11, 14, 11, 10, 19, 19, 15]

print(np.mean(LL_DQN_BL))
print(np.mean(LL_DDQN_BL))
# print(np.mean(LL_RELCLIP))
# print(np.mean(LL_RELCLIP_LOWERMEANRETURN_UPPERMAXRETURN))
# print(np.mean(LL_RELCLIP_LOWER_HALFMAXTD_UPPER_MAXTD))
# print(np.mean(LL_RELCLIP_MAXTDTIMESPROPREL))
# print(np.mean(LL_RELCLIP_MAXTD))
print(np.mean(LL_RELCLIP_SCALEDBOOTSTRAP))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
LL_DDQN_BL = [13, 28, 19, 45, 23, 33, 24, 42, 19, 20]
LL_DQN_BL = [47, 15, 25, 40, 15, 37, 39, 22, 23, 24]
LL_RELCLIP_SCALEDBOOTSTRAP = [21, 17, 12, 11, 14, 11, 10, 19, 19, 15]

data = [LL_DQN_BL, LL_DDQN_BL, LL_RELCLIP_SCALEDBOOTSTRAP]
labels = ['DQN', 'DDQN', 'ReaDQN']
colors = ['black', 'grey', 'red']
alpha = 0.4

# Create the violin plot
fig, ax = plt.subplots(figsize=(8, 6))
parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

# Set violin body colors with alpha
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(alpha)

# Plot horizontal mean lines in solid color
for i, array in enumerate(data):
    mean = np.mean(array)
    ax.hlines(y=mean, xmin=i + 0.75, xmax=i + 1.25, color=colors[i], linewidth=2.5, label=f'{labels[i]} Mean')

# Formatting
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels)
ax.set_ylabel('Value')
ax.set_title('Lunar Lander: Steps until completion')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()








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