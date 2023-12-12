# https://www.youtube.com/watch?v=P9XezMuPfLE&list=PL-9x0_FO_lglnlYextpvu39E7vWzHhtNO&index=1
# https://www.gymlibrary.dev/environments/classic_control/cart_pole/

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import pickle # save models

n_discretize = 10
cart_postion_space        = np.linspace(-2.4, 2.4, n_discretize)
cart_velocity_space       = np.linspace(-4, 4, n_discretize)

pole_angle_space          = np.linspace(-0.20943951, 0.20943951, n_discretize)
pole_angle_velocity_space = np.linspace(-4, 4, n_discretize)

modle_name_pkl = 'cartpole.pkl'

#discritize action space
def get_state(observation):
  cart_postion, cart_velocity, pole_angle, pole_angle_velocity = observation

  # gets the index of the value
  cart_postion_bucket_index  = int(np.digitize(cart_postion, cart_postion_space))
  cart_velocity_bucket_index = int(np.digitize(cart_velocity, cart_velocity_space))

  pole_angle_bucket_index    = int(np.digitize(pole_angle, pole_angle_space))
  pole_angle_velocity_index  = int(np.digitize(pole_angle_velocity, pole_angle_velocity_space))

  return (cart_postion_bucket_index, cart_velocity_bucket_index, \
          pole_angle_bucket_index, pole_angle_velocity_index)


def max_action(Q, state, actions=[0, 1]):
  values = np.array([Q[state, a] for a in actions])
  action = np.argmax(values)

  return action

if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  n_games = 50_000
  alpha = 0.1
  gamma = 0.9
  eps = 1.0

  action_space = [0, 1]

  states = []
  for cart_pos in range(n_discretize + 1):
    for cart_vel in range(n_discretize + 1):
      for pole_angle in range(n_discretize + 1):
        for pole_angle_vel in range(n_discretize + 1):
          states.append((cart_pos, cart_vel, pole_angle, pole_angle_vel))
        

  load = False

  if load == False:
    Q = {}
    for state in states:
      for action in action_space:
        Q[state, action] = 0
  else:
    pickle_in = open('acrobot.pkl', 'rb')
    Q = pickle.load(pickle_in)
    env = wrappers.Monitor(env, "tmp/cartpole", video_callable=lambda episode_id: True, force=True)


  score = 0
  total_rewards = np.zeros(n_games)

  for i in range(n_games):
    obs = env.reset()
    done = False

    if i % 5000 == 0:
      print('episode ', i, ' score ', score, ' eps ', eps)

    score = 0
    state = get_state(obs)

    # action = max_action(Q, state) if np.random.random() > eps else env.action_space.sample()
    action = env.action_space.sample() if np.random.random() < eps else \
                 max_action(Q, state)
    # rand = np.random.random()
    # action = max_action(Q, state) if rand < (1 - eps) else env.action_space.sample()
    # action = maxAction(Q, state) if rand < (1 - eps) else env.action_space.sample()

    while  not done:
      obs_, reward, done, info = env.step(action)
      state_ = get_state(obs_)

      action_ = env.action_space.sample() if np.random.random() < eps else \
                 max_action(Q, state_)
      # rand = np.random.random()
      # action_ = max_action(Q, state_) if rand < (1 - eps) else env.action_space.sample()

      score += reward
      Q[state, action] = Q[state, action] + \
              alpha * (reward + gamma * Q[state_, action_] - Q[state, action])

      state = state_
      action = action_

    total_rewards[i] = score
    eps = eps - 2 / n_games if eps > 0.0 else 0.0

  mean_rewards = np.zeros(n_games)
  for t in range(n_games):
    mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])

  figure, axis = plt.subplots(2, 1)

  axis[0].plot(mean_rewards)
  axis[0].set_title("mean_rewards")

  axis[1].plot(total_rewards)
  axis[1].set_title("total_rewards")

  plt.show()
  figure.savefig('rewards.png')

  f = open(modle_name_pkl, 'wb')
  pickle.dump(Q, f)
  f.close()