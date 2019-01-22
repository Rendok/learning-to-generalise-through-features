"""
A main file coordination interactions between RL and symbolic solver
"""
import pybullet as p
import gym_kuka_multi_blocks.envs.kuka_hrl_env as e

import ray
from ray.tune.registry import register_env
import ray.tune as tune
import argparse


# needs to register a custom environment
def env_creator(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_hrl_env as e

    print("plan: \n", plan)

    env = e.KukaHRLEnv(renders=renders, plan=plan)
    return env

if __name__ == '__main__':
    env = e.KukaHRLEnv(renders=False, plan=[])


    plan, cost, evaluations = env.solve(load_world=env.load_world,
                                        pddlstream_from_problem=env.pddlstream_from_problem,
                                        teleport=True)

    #print("plan: \n", plan)
    #print("cost", cost)
    #print("evaluation", evaluations)

    #env.execute_rl_action(plan[0])
    #env.reset(render=True)

    # register a custom environment
    register_env("KukaHRLEnv-v0", env_creator)

    # assign model variables to commandline arguments
    ray.init()

    # run an experiment with a config
    tune.run_experiments({
        "my_experiment": {
            "run": "PPO",
            "env": "KukaHRLEnv-v0",
            "stop": {"episode_reward_mean": 50},
            "checkpoint_freq": 10,
            "checkpoint_at_end": True,
            "config": {
                "num_gpus": 0,
                "num_workers": 1,
                "num_envs_per_worker": 1,
                "horizon": 20,
                "sample_batch_size": 50,
                "train_batch_size": 2500,
            },
        },
    })

    #observation, reward, _, info = env.step([0, 0, 0, 0])

    #print("OBSERVATION", observation)
    #print("REWARD", reward)
    #print("INFO", info)


    #env.execute(plan)

    #input('Finish?')
