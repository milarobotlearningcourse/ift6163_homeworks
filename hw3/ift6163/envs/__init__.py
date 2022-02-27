from gym.envs.registration import register

def register_envs():
    register(
        id='cheetah-ift6163-v0',
        entry_point='ift6163.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    register(
        id='obstacles-ift6163-v0',
        entry_point='ift6163.envs.obstacles:Obstacles',
        max_episode_steps=500,
    )
    register(
        id='reacher-ift6163-v0',
        entry_point='ift6163.envs.reacher:Reacher7DOFEnv',
        max_episode_steps=500,
    )
