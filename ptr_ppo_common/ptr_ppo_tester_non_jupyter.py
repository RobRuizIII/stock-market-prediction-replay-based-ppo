import numpy as np
import torch as th
import gym
from torch.nn import functional as F
from stable_baselines3 import PPO
from ptr_ppo_common.ptr_ppo import PTR_PPO


if __name__ == "__main__":

# Parallel environments
    # env = gym.make("CartPole-v1")

    # model = PPO(policy = "MlpPolicy", env =  env, verbose=1)
    # model.learn(total_timesteps=25000)

    # obs = env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()



    # Parallel environments
    env = gym.make("CartPole-v1")

    model = PTR_PPO(policy = "MlpPolicy", env =  env, verbose=1)
    model.learn(total_timesteps=25000)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
