import ray
from ray.tune.registry import register_env
import ray.tune as tune
import argparse

# needs to register a custom environment
def env_creator(renders=False):
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
        default=2
    )

    parser.add_argument(
        '--gpu',
        help='The use of gpu',
        type=bool,
        default=True
    )

    parser.add_argument(
        '--run',
        help='Algorithm to run',
        type=str,
        default="PPO"
    )

    args = parser.parse_args()

    # register a custom environment
    register_env("KukaMultiBlocks-v0", env_creator)

    # assign model variables to commandline arguments
    ray.init(redis_address=args.redis_address)

    # run an experiment with a config
    tune.run_experiments({
        "my_experiment": {
            "run": args.run,
            "env": "KukaMultiBlocks-v0",
            "stop": {"episode_reward_mean": 2000},
            "checkpoint_freq": 20,
            "checkpoint_at_end": 1,
            "config": {
                # "gpu": args.gpu,
                "num_gpus": 1,
                "num_workers": args.num_workers,
                "horizon": 1000,
                #"num_envs_per_worker": 4,
            },
        },
    })