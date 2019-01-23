# Kuka arm gym environment with pybullet as a physical engine and pddlstream as a symbolic solver

The solver in use is from https://github.com/caelan/pddlstream

A symbolic solver generates a high level plan, executed by an RL agent. Out-of-box pddlstream action implementations are very basic and will be changed to RL ones. For example, grasp is implemented as sticking one object to another.

*A new environment is in the hrl branch.*

## Installation

```bash
git clone <URL>.git
cd rl-task-planning
pip3 install -e .
```

### Pddlstream

The path to the ```pddlstream``` sources needs to be added to PYTHONPATH, in
bash this would be done with the command

```
export PYTHONPATH=$PYTHONPATH:<path to pddlstream sources>
```

#### How to run the solver
```bash
python3 run.py
```
