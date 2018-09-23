# Gym kuka arm environment with pybullet as a physical engine.

Half-working initial commit

##### Installation

```
cd [gym-kuka-multi-blocks]
pip install -e .
```

##### How to run (sample)

```
import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
env = e.KukaMultiBlocksEnv(renders=True, numObjects=3)

env.reset()

reward = 0
for _ in range(100):
    obs, rew, done, info = env.step(env.action_space.sample())
    reward += rew
    
print(reward)

```