# Kuka arm gym environment with pybullet as a physical engine

The branches represent different chapters of my thesis. The master branch is pretty useless, though. Due to the exploratory nature of the research, the code is rather messy. But to keep the result reproducible, I am not going to clean up the code.

#### Installation

```bash
git clone <URL>.git
cd rl-task-planning
pip3 install -e .
```

#### How to run gym environment (sample)

```python
import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
env = e.KukaMultiBlocksEnv(renders=True, numObjects=3)

env.reset()

reward = 0
for _ in range(100):
    obs, rew, done, info = env.step(env.action_space.sample())
    reward += rew
    
print(reward)

```
#### How to run the solver
```bash
python3 pddl_solver.py
```

#### The simulator
https://pybullet.org/wordpress/

#### Reacher env
https://github.com/benelot/pybullet-gym

### PPO implementation
https://docs.ray.io/en/latest/rllib.html#
