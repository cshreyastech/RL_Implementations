import numpy as np
import gym

if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  for i in range(10):
    obs = env.reset()
    done = False

    while  not done:
      action = env.action_space.sample()
      print(obs)
      print('action', action)
      obs, reward, done, info = env.step(action)
      env.render(mode="human")

    env.close()