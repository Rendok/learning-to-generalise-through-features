import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

c1 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-07_01-53-28pb9qko3w-tag-ray_tune_episode_reward_mean.csv')
c2 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-08_05-03-03rysgn407-tag-ray_tune_episode_reward_mean.csv')
c3 = pd.read_csv('/Users/dgrebenyuk/Research/rl-task-planning/csv/run_PPO_KukaMultiBlocks-v0_0_2019-05-08_07-03-38kr507iig-tag-ray_tune_episode_reward_mean.csv')

print(c1.head())
sns.lineplot('Step', 'Value', data=c1)
sns.lineplot('Step', 'Value', data=c2)
sns.lineplot('Step', 'Value', data=c3)
plt.title('Learning Rates')
plt.legend(title='Experiment', loc='lower right', labels=['1', '2', '3'])
plt.show()
