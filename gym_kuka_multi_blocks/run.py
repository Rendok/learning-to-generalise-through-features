"""
A main file coordination interactions between RL and symbolic solver
"""
import pybullet as p
import gym_kuka_multi_blocks.envs.kuka_hrl_env as e

if __name__ == '__main__':
    env = e.KukaHRLEnv(renders=False,
                       num_objects=2)

    env.reset()

    plan, cost, evaluations = env.solve(teleport=True)

    print("plan", plan)
    print("cost", cost)
    print("evaluation", evaluations)

    env.execute(plan)

    #input('Finish?')
