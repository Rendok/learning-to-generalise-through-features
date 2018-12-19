# Kuka arm gym environment with pybullet as a physical engine and pddlstream as a symbolic solver

The solver in use is from https://github.com/caelan/pddlstream

A symbolic solver generates a high level plan, executed by an RL agent. Out-of-box pddlstream action implementations are very basic and will be changed to RL ones. For example, grasp is implemented as sticking one object to another.

_I am in proccess of merging pddl and gym envirionment_

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
