import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from torch.nn import functional as F
import torch as th
from scipy.special import rel_entr

x = np.arange(100) / 100

print(x ** .2)
print(x ** .1)

# phi = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
# t = 1

# phi_t = phi[t]

# same_episode_as_t = phi == phi_t

# print(same_episode_as_t)
# gives [ True  True  True  True False False False False False False False False]

"""
PER =  22 + 33 + 24 + 12 + 24 + 20 + 17 + 24 + 21 + 22
a4  =  15 + 28 + 23 + 20 + 10 + 19 + 24 + 13 + 26 + 16
a5  =  15 + 13 + 31 + 27 + 11 + 15 + 21 + 30 + 20 + 24
a6  =  18 + 23 +  2 + 19 + 10 + 19 + 25 + 20 + 24 + 25
a6a0 = 10 +  2 + 21 + 25 + 11 + 20 + 20 + 23 + 25 + 26
a6a1 = 37 + 16 + 27 + 33 + 27 + 26 + 29 + 25 + 25 + 17
a6a3 = 7 + 16 + 24 + 30 + 8 + 27 + 21 + 23 + 28 + 23
a6RR = 19 +  2 + 22 + 21 + 14 + 18 + 10 + 11 + 22 + 21
uni  = 21 + 40 + 53 + 57 + 10 + 61 + 57 + 42 + 40 + 50 
print(f"{PER, a4, a5, a6, a6a0, a6a1, a6a3, a6RR, uni=}")

sys.exit()
"""























# td1 = abs(np.random.normal(scale=50, size=(int(1e6) - 10_000,)))
# td2 = abs(np.random.normal(scale=1000, size=(10_000,)))
# td = np.append(td1,td2)

# # plt.hist(td, bins=100)
# # plt.show()
# # sys.exit()

# rel = np.random.uniform(low=0, high=1, size=(int(1e6),))
# alpha = 0.6
# per_w = td ** alpha
# per_p = per_w / per_w.sum()

# sim, x, y = [], [], []
# for x_ in np.arange(2, 10)/10:
#     #y_ = x_
#     for y_ in np.arange(2, 10)/10:
#         reaper_w = td**x_ * rel**y_
#         reaper_p = reaper_w / reaper_w.sum()

#         sim_ = np.sum(rel_entr(per_p, reaper_p))

#         x.append(x_)
#         y.append(y_)
#         sim.append(sim_)

# res = np.vstack([x,y,sim]).T
# res = res[res[:, 2].argsort()]
# print(res[:20, :])


# # # LL
# # per = 161.4
# # tper = 187.3
# # rd  = -180.07


# # # CP
# # per = 161.4
# # tper = 187.3
# # rd  = 

# # tper_ = tper - rd
# # per_ = per - rd
# # print( (tper_ - per_) / per_ * 100)

# # sys.exit()

# # --------------------------------------------------------------------------------------------------------------------------------

# # sys.exit()
# td = np.random.uniform(size=(int(1e6),)) + 1e-6 # np.array([.1, .5, .9, .1, .5, .9, .1, .5, .9])
# td = abs(np.random.normal(scale=50, size=(int(1e6))))
# td1 = abs(np.random.normal(scale=50, size=(int(1e6) - 10_000,)))
# td2 = abs(np.random.normal(scale=1000, size=(10_000,)))
# td = np.append(td1,td2)
# rel = np.random.uniform(size=(int(1e6),)) # np.array([.1, .1, .1, .5, .5, .5, .9, .9, .9])

# plt.rcParams.update({'font.size': 6})

# alphas1 = [0, .3, .4, .5, .6, .7, .8]# (np.array(np.arange(11, step=2))/10)# [1:-1]
# alphas2 = [0, .3, .4, .5, .6]# (np.array(np.arange(11, step=2))/10)# [1:-1]

# fig, axs = plt.subplots(ncols=len(alphas2), nrows=len(alphas1))
# for idx1, alpha1 in enumerate(alphas1):
#     for idx2, alpha2 in enumerate(alphas2):
#         w = td**alpha1 * rel**alpha2
#         p = w / w.sum()
#         axs[idx1, idx2].hist(p, bins=100)
#         axs[idx1, idx2].set_title(f"Alpha1 = {alpha1}, Alpha2 = {alpha2}")

# plt.tight_layout()
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------------------


# # [0.06309573 0.1657227  0.23580093
# # 0.1657227  0.43527528 0.61933769
# # 0.23580093 0.61933769 0.88123353]


# # baseline_noclip = {'Alien': 200.0, 'Amidar': 0.6, 'Assault': 226.8, 'Asterix': 190.0, 'Asteroids': 532.0, 'Atlantis': 15120.0, 'BankHeist': 16.0, 'BattleZone': 2800.0, 'BeamRider': 404.8, 'Berzerk': 80.0, 'Bowling': 24.0, 'Boxing': 2.6, 'Breakout': 0.8, 'Centipede': 2027.2, 'ChopperCommand': 800.0, 'CrazyClimber': 6660.0, 'Defender': 2700.0, 'DemonAttack': 160.0, 'DoubleDunk': -15.2, 'Enduro': 0.0, 'FishingDerby': -95.8, 'Freeway': 0.0, 'Frostbite': 80.0, 'Gopher': 352.0, 'Gravitar': 100.0, 'Hero': 449.0, 'IceHockey': -10.2, 'Jamesbond': 30.0, 'JourneyEscape': -28040.0, 'Kangaroo': 40.0, 'Krull': 1880.0, 'KungFuMaster': 220.0, 'MontezumaRevenge': 0.0, 'MsPacman': 338.0, 'NameThisGame': 1710.0, 'Phoenix': 356.0, 'Pitfall': -311.0, 'Pong': -20.2, 'PrivateEye': 80.0, 'Qbert': 180.0, 'Riverraid': 1428.0, 'RoadRunner': 0.0, 'Robotank': 1.2, 'Seaquest': 56.0, 'Skiing': -16314.0, 'Solaris': 1552.0, 'SpaceInvaders': 176.0, 'StarGunner': 520.0, 'Tennis': -24.0, 'TimePilot': 3700.0, 'Tutankham': 4.6, 'UpNDown': 322.0, 'Venture': 0.0, 'VideoPinball': 27080.6, 'WizardOfWor': 320.0, 'YarsRevenge': 3243.0, 'Zaxxon': 40.0}
# # baseline_clip = {'Alien': 200.0, 'Amidar': 0.6, 'Assault': 226.8, 'Asterix': 190.0, 'Asteroids': 532.0, 'Atlantis': 15120.0, 'BankHeist': 16.0, 'BattleZone': 2800.0, 'BeamRider': 404.8, 'Berzerk': 80.0, 'Bowling': 24.0, 'Boxing': 2.6, 'Breakout': 0.8, 'Centipede': 2027.2, 'ChopperCommand': 800.0, 'CrazyClimber': 6660.0, 'Defender': 2700.0, 'DemonAttack': 160.0, 'DoubleDunk': -15.2, 'Enduro': 0.0, 'FishingDerby': -95.8, 'Freeway': 0.0, 'Frostbite': 80.0, 'Gopher': 352.0, 'Gravitar': 100.0, 'Hero': 449.0, 'IceHockey': -10.2, 'Jamesbond': 30.0, 'JourneyEscape': -28040.0, 'Kangaroo': 40.0, 'Krull': 1880.0, 'KungFuMaster': 220.0, 'MontezumaRevenge': 0.0, 'MsPacman': 338.0, 'NameThisGame': 1710.0, 'Phoenix': 356.0, 'Pitfall': -311.0, 'Pong': -20.2, 'PrivateEye': 80.0, 'Qbert': 180.0, 'RiverRaid': 1428.0, 'RoadRunner': 0.0, 'Robotank': 1.2, 'Seaquest': 56.0, 'Skiing': -16314.0, 'Solaris': 1552.0, 'SpaceInvaders': 176.0, 'StarGunner': 520.0, 'Tennis': -24.0, 'TimePilot': 3700.0, 'Tutankham': 4.6, 'UpNDown': 322.0, 'Venture': 0.0, 'VideoPinball': 27080.6, 'WizardOfWor': 320.0, 'YarsRevenge': 3243.0, 'Zaxxon': 40.0}


# # MaxSS = 1.
# # RWs = .7, .8, .9, 1.

# # td_errors = np.array([.4, .3, .2, .1])
# # episodes = np.array([1,1,1,1])
# # alpha2 = 1.

# # def get_rel_weights(td_errors, episodes):

# #     # Convert inputs to numpy arrays
# #     td_errors = np.asarray(td_errors, dtype=np.float64)
# #     episodes = np.asarray(episodes)

# #     # Identify unique episodes and their starting indices
# #     unique_episodes, inverse_indices, episode_counts = np.unique(episodes, return_inverse=True, return_counts=True)

# #     # Compute per-episode cumulative sums
# #     cumsum_all = np.cumsum(td_errors)
# #     episode_start_indices = np.r_[0, np.cumsum(episode_counts)[:-1]]  # Start index of each episode

# #     # Compute td_cums: cumulative sum per episode
# #     episode_start_cumsum = cumsum_all[(episode_start_indices-1)[1:]]
# #     episode_start_cumsum = np.insert(episode_start_cumsum, 0, 0)

# #     td_cums = cumsum_all - episode_start_cumsum[inverse_indices]

# #     # Compute td_sums: total sum for each episode (mapped to original shape)
# #     total_sums = np.bincount(inverse_indices, weights=td_errors)
# #     td_sums = total_sums[inverse_indices]

# #     # Compute subsequent TD errors
# #     subsequent_tds = td_sums - td_cums
# #     max_subsequent_td = np.max(subsequent_tds)

# #     # Compute relative weights
# #     rel_weight = (1 - (subsequent_tds / max_subsequent_td) + 1e-6) ** alpha2

# #     return rel_weight

# # print(get_rel_weights(td_errors, episodes))