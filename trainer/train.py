import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import ddpg
import argparse


def env_creator(renders=False):
    import gym
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=3,
                               removeHeightHack=True,
                               isDiscrete=False)
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--redis_address',
        help='Redis IP:port',
        type=str,
        required=True
    )

    parser.add_argument(
        '--num_workers',
        help='The number of workers',
        type=int,
        default=4
    )

    parser.add_argument(
        '--gpu',
        help='The use of gpu',
        type=bool,
        default=True
    )

    args = parser.parse_args()

    # register a custom environment
    register_env("my_env", env_creator)

    # copy, print, and amend the default config
    config = ddpg.apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    #config["num_gpus"] = 0
    config["num_workers"] = args.num_workers
    config["horizon"] = 1000
    #config["train_batch_size"] = 500
    config["num_envs_per_worker"] = 4
    config["gpu"] = args.gpu
    config["num_gpus_per_worker"] = 1/4
    print(config)

    # Assign model variables to commandline arguments
    ray.init(redis_address=args.redis_address)

    # initialize an agent
    agent = ddpg.apex.ApexDDPGAgent(config=config, env="my_env")

    for i in range(10001):
        # Perform one iteration of training
        result = agent.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)
