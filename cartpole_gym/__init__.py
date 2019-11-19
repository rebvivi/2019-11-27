from gym.envs.registration import register

register(
    id='CartPoleEnv11-v0',
    entry_point='cartpole_gym.envs:CartPoleEnv',
    max_episode_steps=2000,
    reward_threshold=195.0,
)


