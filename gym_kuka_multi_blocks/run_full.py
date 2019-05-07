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
                               numObjects=2,
                               isDiscrete=False,
                               isTest=-1,
                               maxSteps=20,
                               actionRepeat=80,
                               operation='move_pick')
    return env


def env_creator_place(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=2,
                               isDiscrete=False,
                               isTest=-1,
                               maxSteps=20,
                               actionRepeat=80,
                               operation='place')
    return env


def init_move_pick(renders=False):

    register_env("pick", env_creator_move_pick)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1

    env = env_creator_move_pick(renders=renders)

    agent = ppo.PPOAgent(config=config, env="pick")
    agent.restore("/Users/dgrebenyuk/ray_results/pick/PPO_KukaMultiBlocks-v0_0_2019-03-27_11-13-30nbdyzah7/checkpoint_300/checkpoint-300")

    return agent, env


def init_place(renders=False):

    register_env("place", env_creator_place)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1

    env = env_creator_place(renders=renders)

    agent = ppo.PPOAgent(config=config, env="place")
    agent.restore("/Users/dgrebenyuk/ray_results/place/PPO_KukaMultiBlocks-v0_0_2019-04-03_04-32-48qsc01eeg/checkpoint_100/checkpoint-100")
    #agent.restore("/Users/dgrebenyuk/ray_results/place/PPO_KukaMultiBlocks-v0_0_2019-04-03_09-59-16z2_syfpz/checkpoint_120/checkpoint-120")

    return agent, env


def execute(plan, viewer=False, display=True, simulate=False, teleport=False):

    if plan is None:
        raise TypeError

    ray.init()

    # create an environment
    env = env_creator_pick(renders=True)

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
        body = params[1].body
        pose = params[1].pose
        print('_____SOLVER_____', name, body, pose)
        env.set_goal(body, 'pick')
        rl_loop(env, obs, policies[0][1])
    elif name == 'place':
        body = params[1].body
        pose = params[1].pose
        pose[0][2] -= 0.7
        #print(params[2].grasp_pose, type(params[2]))
        print('_____SOLVER_____', name, body, pose)
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

    plan, cost, evaluations = solve(load_world=load_world,
                                    pddlstream_from_problem=pddlstream_from_problem,
                                    teleport=True
                                    )

    #print("plan: \n", plan)
    #print("cost", cost)
    #print("evaluation", evaluations)

    execute(plan)
