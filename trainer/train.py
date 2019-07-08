import ray
from ray.tune.registry import register_env
import ray.tune as tune
import argparse

my_experiment = 'move_pick'
# my_experiment = 'place'

# needs to register a custom environment
def env_creator(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=4,
                               isTest=10,
                               operation=my_experiment,
                               constantVector=False,
                               blocksInObservation=True,
                               sensing=True,
                               num_sectors=(16, 8)
                               )
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--redis_address',
        help='Redis "IP:port"',
        type=str,
        required=True
    )

    parser.add_argument(
        '--num_workers',
        help='The number of workers',
        type=int,
        default=1
    )

    parser.add_argument(
        '--gpu',
        help='The use of gpu in DDPG, DQN, and APEX',
        type=str,
        default="False"
    )

    parser.add_argument(
        '--num_gpus',
        help='The number of gpus in PPO',
        type=int,
        default=0
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

    parser.add_argument(
        '--num_envs_per_worker',
        help='The number of_environments per worker',
        type=int,
        default=1
    )

    parser.add_argument(
        '--checkpoint_freq',
        help='Checkpoint frequency',
        type=int,
        default=50
    )

    args = parser.parse_args()

    if args.gpu == "True":
        gpu = True
    elif args.gpu == "False":
        gpu = False
    else:
        raise ValueError

    # register a custom environment
    register_env("KukaMultiBlocks-v0", env_creator)

    # assign model variables to commandline arguments
    ray.init(redis_address=args.redis_address)

    if args.run == "DDPG" or args.run == "DQN":

        # run an experiment with a config
        tune.run_experiments({
            my_experiment: {
                "run": args.run,
                "env": "KukaMultiBlocks-v0",
                "stop": {"episode_reward_mean": 50},
                "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_at_end": args.checkpoint_at_end,
                "config": {
                    "gpu": gpu,  # ddpg
                    "num_workers": args.num_workers,
                    "num_envs_per_worker": args.num_envs_per_worker,
                    "horizon": 20,
                    "optimizer_class": "AsyncReplayOptimizer",
                },
            },
        })
    elif args.run == "APEX_DDPG" or args.run == "APEX":

        # run an experiment with a config
        tune.run_experiments({
            my_experiment: {
                "run": args.run,
                "env": "KukaMultiBlocks-v0",
                "stop": {"episode_reward_mean": 50},
                "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_at_end": args.checkpoint_at_end,
                "config": {
                    "gpu": gpu,  # ddpg
                    "num_workers": args.num_workers,
                    "num_envs_per_worker": args.num_envs_per_worker,
                    "horizon": 20,
                },
            },
        })
    else:
        # run an experiment with a config
        tune.run_experiments({
            my_experiment: {
                "run": args.run,
                "env": "KukaMultiBlocks-v0",
                "stop": {"episode_reward_mean": 50},
                "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_at_end": args.checkpoint_at_end,
                #"restore": "/home/ubuntu/ray_results/move_pick/PPO_KukaMultiBlocks-v0_0_2019-06-19_07-25-47zbzvryjg/checkpoint_620/checkpoint-620",
                #"resume": True,
                    "config": {
                    "num_gpus": args.num_gpus,  # ppo
                    "num_workers": args.num_workers,
                    "num_envs_per_worker": args.num_envs_per_worker,
                    "horizon": 40,
                    "sample_batch_size": 25,  # 50,
                    "train_batch_size": 1250,  # 2500,
                },
            },
        })
