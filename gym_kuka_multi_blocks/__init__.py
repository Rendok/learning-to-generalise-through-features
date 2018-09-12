import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------


register(
    id='KukaMultiBlocks-v0',
    entry_point='gym_kuka_multi_blocks.envs:KukaMultiBlocksEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)

register(
    id='KukaCamMultiBlocks-v0',
    entry_point='gym_kuka_multi_blocks.envs:KukaCamMultiBlocksEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)


def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
    return btenvs
