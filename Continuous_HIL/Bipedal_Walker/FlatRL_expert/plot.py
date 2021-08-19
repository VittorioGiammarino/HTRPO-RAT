
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

with open('results/mean_TD3_BipedalWalker-v3_Nseed_20.npy', 'rb') as f:
    TD3_mean = np.load(f, allow_pickle=True)
    
with open('results/std_TD3_BipedalWalker-v3_Nseed_20.npy', 'rb') as f:
    TD3_std = np.load(f, allow_pickle=True)
    
with open('results/steps_TD3_BipedalWalker-v3_Nseed_20.npy', 'rb') as f:
    steps = np.load(f, allow_pickle=True)
    
with open('results/evaluation_SAC_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    SAC_mean = np.load(f, allow_pickle=True)
    
with open('results/mean_SAC+GAIL_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    SAC_GAIL_mean = np.load(f, allow_pickle=True)

with open('results/evaluation_TRPO_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    TRPO_mean = np.load(f, allow_pickle=True)
    
with open('results/evaluation_TRPO+GAIL_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    TRPO_GAIL_mean = np.load(f, allow_pickle=True)
    
with open('results/evaluation_UATRPO_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    UATRPO_mean = np.load(f, allow_pickle=True)
    
with open('results/evaluation_PPO_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    PPO_mean = np.load(f, allow_pickle=True)
    
# %%

fig, ax = plt.subplots()
# plt.xscale('log')
# plt.xticks(Samples, labels=['100', '200', '500', '1k', '2k'])
clrs = sns.color_palette("husl", 6)
ax.plot(steps, TD3_mean, label='TD3', c=clrs[0])
ax.fill_between(steps, TD3_mean-TD3_std, TD3_mean+TD3_std, alpha=0.1, facecolor=clrs[0])
ax.plot(steps, SAC_mean, label='SAC', c=clrs[1])
ax.plot(steps, UATRPO_mean, label='UATRPO', c=clrs[2])
ax.plot(steps, TRPO_mean, label='TRPO', c=clrs[3])
ax.plot(steps, PPO_mean, label='PPO', c=clrs[4])
ax.plot(steps, TRPO_GAIL_mean, label='TRPO+GAIL', c=clrs[5])
ax.legend(loc=0, facecolor = '#d8dcd6')
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Bipedal Walker')
