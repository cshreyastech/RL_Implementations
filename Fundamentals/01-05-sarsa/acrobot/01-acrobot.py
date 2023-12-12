# https://www.youtube.com/watch?v=gJZwXkXDFEI&list=PL-9x0_FO_lglnlYextpvu39E7vWzHhtNO&index=
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/acrobot.py
# https://www.gymlibrary.dev/environments/classic_control/acrobot/

import numpy as np
import gym

if __name__ == '__main__':
  env = gym.make('Acrobot-v1')
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