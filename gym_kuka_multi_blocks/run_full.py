"""
A main file coordination interactions between RL and symbolic solver
"""
import pybullet as p
import gym_kuka_multi_blocks.envs.kuka_hrl_env as e
from ray.rllib.agents import ppo

import ray
from ray.tune.registry import register_env

from gym_kuka_multi_blocks.envs.solver import solve, load_world, pddlstream_from_problem


# TODO: add all the trained envs
def env_creator_move_pick(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=4,
                               isTest=10,
                               operation='move_pick',
                               constantVector=False,
                               blocksInObservation=True,  # F - e1, T - e2 or e3
                               sensing=True,
                               num_sectors=(16, 8),
                               globalGripper=True
                               )
    return env


def env_creator_place(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=4,
                               isTest=10,
                               operation='place',
                               constantVector=False,
                               blocksInObservation=True,  # F - e1, T - e2 or e3
                               sensing=True,
                               num_sectors=(16, 8),
                               globalGripper=False
                               )
    return env


def init_move_pick(renders=False):

    register_env("pick", env_creator_move_pick)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1

    env = env_creator_move_pick(renders=renders)

    agent = ppo.PPOAgent(config=config, env="pick")
    agent.restore("/Users/dgrebenyuk/Research/policies/move_pick/test10_4bl_noGr_L0/PPO_KukaMultiBlocks-v0_0_2019-08-10_11-14-24pd7tvr6t/checkpoint_1500/checkpoint-1500")

    return agent, env


def init_place(renders=False):

    register_env("place", env_creator_place)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1

    env = env_creator_place(renders=renders)

    agent = ppo.PPOAgent(config=config, env="place")
    agent.restore("/Users/dgrebenyuk/Research/policies/place/test13_s16_8_5bl_noGr_L0/PPO_KukaMultiBlocks-v0_0_2019-08-09_06-29-56un4wmget/checkpoint_1000/checkpoint-1000")  # 1000

    return agent, env


def execute(plan, viewer=False, display=True, simulate=False, teleport=False):

    if plan is None:
        raise TypeError

    ray.init()

    # create an environment
    env = env_creator_move_pick(renders=True)

    # load policies
    agent1, _ = init_move_pick(renders=False)
    agent2, _ = init_place(renders=False)

    policies = [('pick', agent1), ('place', agent2)]

    obs = env.reset()

    for plan_action in plan:
        execute_rl_action(env, plan_action, obs, policies)


def execute_rl_action(env, plan_action, obs, policies):
    """
    Used to switch to a appropriate RL for the current action
    :param plan_action:
    :return:
    """

    name, params = plan_action

    if name == 'move_free' or name == 'move_holding':
        pass
    elif name == 'pick':
        body = params # params[1].body
        # pose = params[1].pose
        print('_____SOLVER_____', name, body) #, pose)
        env.set_goal(body, 'pick')
        rl_loop(env, obs, policies[0][1])
    elif name == 'place':
        # body = params[1].body
        pose = params # params[1].pose
        # pose[0][2] -= 0.7
        #print(params[2].grasp_pose, type(params[2]))
        print('_____SOLVER_____', name, pose) # body, pose)
        env.set_goal(pose, 'place')
        rl_loop(env, obs, policies[1][1])
    else:
        pass


def rl_loop(env, obs, policy):
    done = False
    while not done:
        action = policy.compute_action(obs)
        obs, rew, done, info = env.step(action)
        # obs, rew, done, info = env.step([0, 0, -1, 0.1])
        print("__________REWARD____________", rew, info)


if __name__ == '__main__':

    # plan, cost, evaluations = solve(load_world=load_world,
    #                                 pddlstream_from_problem=pddlstream_from_problem,
    #                                 teleport=True
    #                                 )

    plan =[('pick', 3), ('place', [0.4, 0, 0])]
    #print("plan: \n", plan)
    #print("cost", cost)
    #print("evaluation", evaluations)

    execute(plan)
