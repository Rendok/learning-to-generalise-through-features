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
def env_creator_pick(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=2,
                               isDiscrete=False,
                               isTest=-1,
                               maxSteps=20,
                               actionRepeat=80,
                               blockRandom=0.8,
                               operation='pick')
    return env


def env_creator_place(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=2,
                               isDiscrete=False,
                               isTest=-1,
                               maxSteps=20,
                               actionRepeat=80,
                               blockRandom=0.8,
                               operation='place')
    return env


def init_pick(renders=False):
    from ray.rllib.models import ModelCatalog

    register_env("pick", env_creator_pick)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1

    env = env_creator_pick(renders=renders)

    agent = ppo.PPOAgent(config=config, env="pick")
    agent.restore("/Users/dgrebenyuk/ray_results/pick/PPO_KukaMultiBlocks-v0_0_2019-03-25_04-03-30oglxq3k4/checkpoint_580/checkpoint-580")
    #agent.restore("/Users/dgrebenyuk/ray_results/pick/PPO_KukaMultiBlocks-v0_0_2019-03-22_08-58-29dhogmp4r/checkpoint-180")

    return agent, env


def init_place(renders=False):
    from ray.rllib.models import ModelCatalog

    register_env("place", env_creator_place)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1

    env = env_creator_place(renders=renders)

    agent = ppo.PPOAgent(config=config, env="place")
    agent.restore("/Users/dgrebenyuk/ray_results/place/PPO_KukaMultiBlocks-v0_0_2019-03-23_08-35-409p66bqi7/checkpoint_160/checkpoint-160")

    return agent, env


def execute(plan, viewer=False, display=True, simulate=False, teleport=False):

    if plan is None:
        raise TypeError

    ray.init()

    # create an environment
    env = env_creator_pick(renders=True)

    # load policies
    agent1, _ = init_pick(renders=False)
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
        print(name, body, pose)
        env.set_goal(body, 'pick')
        rl_loop(env, obs, body, pose, policies[0][1])
    elif name == 'place':
        body = params[1].body
        pose = params[1].pose
        print(name, body, pose)
        env.set_goal(pose[0], 'place')
        rl_loop(env, obs, body, pose, policies[1][1])
    else:
        pass


def rl_loop(env, obs, body, pose, policy):
    # TODO: construct a goal from 'body' and 'pose'
    # TODO: update the goal
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


    # register a custom environment
    # TODO: add all the trained envs
    #register_env("KukaHRLEnv-v0", env_creator)

    #ray.init()

    execute(plan)
