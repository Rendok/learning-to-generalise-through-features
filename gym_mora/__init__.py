from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='MultiSlide{}-v0'.format(suffix),
        entry_point='gym_mora.envs:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiPickAndPlace{}-v0'.format(suffix),
        entry_point='gym_mora.envs:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiReach{}-v0'.format(suffix),
        entry_point='gym_mora.envs:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MultiPush{}-v0'.format(suffix),
        entry_point='gym_mora.envs:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )


