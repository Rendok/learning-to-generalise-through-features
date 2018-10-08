import ray
from ray.tune.registry import register_env
import ray.tune as tune
import argparse

# needs to register a custom environment
def env_creator(renders=False):
    import pybullet_envs
    import gym
    env = gym.make("AntBulletEnv-v0")
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
        default=3
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

    parser.add_argument(
        '--checkpoint_at_end',
        help='Save the results after finishing',
        type=bool,
        default=1
    )

    args = parser.parse_args()

    # register a custom environment
    register_env("AntBulletEnv-v0", env_creator)

    # assign model variables to commandline arguments
    ray.init(redis_address=args.redis_address)

    # run an experiment with a config
    tune.run_experiments({
        "my_experiment": {
            "run": args.run,
            "env": "AntBulletEnv-v0",
            "stop": {"episode_reward_mean": 2000},
            "checkpoint_freq": 100,
            "checkpoint_at_end": args.checkpoint_at_end,
            "config": {
                # "gpu": args.gpu,
                "num_gpus": 0,
                "num_workers": args.num_workers,
                #"horizon": 1000,
                #"num_envs_per_worker": 4,
            },
        },
    })