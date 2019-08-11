import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# No blocks in obs nor in reward
# c1 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test10_4bl_L0_PPO_KukaMultiBlocks-v0_0_2019-06-19_11-29-17ythlnk10-tag-ray_tune_episode_reward_mean.csv')
c1 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test12_5bl_L0_PPO_KukaMultiBlocks-v0_0_2019-06-25_10-10-56gxsemph4-tag-ray_tune_episode_reward_mean.csv')
# blocks in obs
# c2 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test10_e1_4bl_L0_PPO_KukaMultiBlocks-v0_0_2019-06-27_07-30-077i_1glku-tag-ray_tune_episode_reward_mean.csv')
c2 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test12_e1_5bl_L0_PPO_KukaMultiBlocks-v0_0_2019-07-03_02-26-41y8thz8zy-tag-ray_tune_episode_reward_mean.csv')
# blocks in obs and reward L = 1
# c3 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test10_s16_8_4bl_L0_PPO_KukaMultiBlocks-v0_0_2019-07-29_05-46-38y250zip7-tag-ray_tune_episode_reward_mean.csv')
c3 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test12_s16_8_5bl_L0_PPO_KukaMultiBlocks-v0_0_2019-07-17_03-38-44d0nyfysk-tag-ray_tune_episode_reward_mean.csv')
# blocks in obs and reward L = 0.5
c4 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_test13_s16_8_5bl_L0_PPO_KukaMultiBlocks-v0_0_2019-07-22_04-32-20gtbab48l-tag-ray_tune_episode_reward_mean.csv')

# Shifting reward lambda = 0
# c4 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/retrained0/run_test10_s16_8_4bl_L0_PPO_KukaMultiBlocks-v0_0_2019-07-29_05-46-38y250zip7-tag-ray_tune_episode_reward_mean.csv')
# lambda = 0.5
# c5 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/retrained0/run_PPO_KukaMultiBlocks-v0_0_2019-05-21_07-13-500k_fzofj-tag-ray_tune_episode_reward_mean.csv')
# lambda = 1
# c6 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/retrained0/run_PPO_KukaMultiBlocks-v0_0_2019-05-22_05-20-56eaerq7wv-tag-ray_tune_episode_reward_mean.csv')

# c2.loc[:, 'Step'] /= 8
# c3.loc[:, 'Step'] /= 8
# c1.loc[:, 'Step'] /= 8
# c4.loc[:, 'Step'] /= 8
# c5.loc[:, 'Step'] /= 8
# c6.loc[:, 'Step'] /= 8

# c3['Mean'] = savgol_filter(c3['Value'], 101, 3)

# c3 = pd.melt(c3, id_vars=['Step'],
#              value_vars=['Value', 'Mean'])

# print(c3.head())
g = sns.lineplot('Step', 'Value', data=c1)
sns.lineplot('Step', 'Value', data=c2)
#sns.lineplot('Step', 'Value', data=c3, alpha=0.3, c="g")
sns.lineplot('Step', 'Value', data=c3)

sns.lineplot('Step', 'Value', data=c4)
# sns.lineplot('Step', 'Value', data=c5)
# sns.lineplot('Step', 'Value', data=c6)

#plt.title('Learning Rates')
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
# plt.legend(title='Experiment', loc='lower right', labels=['$\lambda = 0$', '$\lambda = 1/16$'])
plt.legend(title='Experiment Name', loc='lower right', labels=['all in obs', 'none in obs', 'sens(16, 8)', 'sens(16, 8) ris'])


x = c1['Step'].values
xlabels = ['{:,.0f}'.format(x) + 'k' for x in g.get_xticks()/1000]

g.set_xticklabels(xlabels)
#plt.xscale('log')

plt.show()
