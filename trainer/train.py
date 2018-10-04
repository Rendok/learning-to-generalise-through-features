import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import ddpg


def env_creator(renders=False):
    import gym
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=3,
                               removeHeightHack=True,
                               isDiscrete=False)
    return env




register_env("my_env", env_creator)

config = ddpg.apex.APEX_DDPG_DEFAULT_CONFIG.copy()
print(config)
#config["num_gpus"] = 0
config["num_workers"] = 4
config["horizon"] = 1000
#config["train_batch_size"] = 500
config["num_envs_per_worker"] = 4

ray.init()

agent = ddpg.apex.ApexDDPGAgent(config=config, env="my_env")

for i in range(10001):
    # Perform one iteration of training
    result = agent.train()

    if i % 20 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)
        print(pretty_print(result))
