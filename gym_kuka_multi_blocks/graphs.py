import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# No blocks in obs nor in reward
#c1 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-07_01-53-28pb9qko3w-tag-ray_tune_episode_reward_mean.csv')
# blocks in obs
c2 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-08_05-03-03rysgn407-tag-ray_tune_episode_reward_mean.csv')
# blocks in obs and reward L = 1
c3 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-08_07-03-38kr507iig-tag-ray_tune_episode_reward_mean.csv')
# blocks in obs and reward L = 0.5
c1 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-21_05-55-10gm1gv_u0-tag-ray_tune_episode_reward_mean.csv')

# Shifting reward lambda = 0
c4 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/retrained0/run_PPO_KukaMultiBlocks-v0_0_2019-05-21_06-43-15_jtk6qv4-tag-ray_tune_episode_reward_mean.csv')
# lambda = 0.5
c5 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/retrained0/run_PPO_KukaMultiBlocks-v0_0_2019-05-21_07-13-500k_fzofj-tag-ray_tune_episode_reward_mean.csv')
# lambda = 1
c6 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/retrained0/run_PPO_KukaMultiBlocks-v0_0_2019-05-22_05-20-56eaerq7wv-tag-ray_tune_episode_reward_mean.csv')

c2.loc[:, 'Step'] /= 8
c3.loc[:, 'Step'] /= 8
c1.loc[:, 'Step'] /= 8
c4.loc[:, 'Step'] /= 8
c5.loc[:, 'Step'] /= 8
c6.loc[:, 'Step'] /= 8

c3['Mean'] = savgol_filter(c3['Value'], 101, 3)

c3 = pd.melt(c3, id_vars=['Step'],
             value_vars=['Value', 'Mean'])

print(c3.head())
g = sns.lineplot('Step', 'Value', data=c2)
sns.lineplot('Step', 'Value', data=c1)
#sns.lineplot('Step', 'Value', data=c3, alpha=0.3, c="g")
sns.lineplot('Step', 'value', data=c3)

sns.lineplot('Step', 'Value', data=c4)
sns.lineplot('Step', 'Value', data=c5)
sns.lineplot('Step', 'Value', data=c6)

plt.title('Learning Rates')
plt.xlabel('Episodes')
plt.legend(title='Experiment', loc='lower right', labels=['lambda = 0', 'lambda = 0.5', 'lambda = 1',
                                                                        'lambda = 0', 'lambda = 0.5', 'lambda = 1'])

x = c2['Step'].values
xlabels = ['{:,.0f}'.format(x) + 'k' for x in g.get_xticks()/1000]

g.set_xticklabels(xlabels)
#plt.xscale('log')

plt.show()
