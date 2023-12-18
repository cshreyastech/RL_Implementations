"""
Double Q Learning
https://www.youtube.com/watch?v=Q99bEPStnxk&list=PL-9x0_FO_lglnlYextpvu39E7vWzHhtNO&index=3
- Q learning is a model free bootstrapped off policy learning
  - Model free: Doesnt need complete state transition dynamics of the environment.
    It learns by playing it.
  - bootstrapped: It donent need much help to get started. It generates estimates using
    the initial estimates which are totally arbitrary except for the terminal states
  - off policy: Uses a separate. Uses a behavior policy and target policy to learn
  about the environment and generate behavior.
- Maximization bias: Happens because of using same Q to maximizing as well as value of the action.
  The soution is to use two seperate Qs to determine max action and the value. Set a relation between
  them and alternate them between the game. This will eleminate bias over time.


Algorithm:
 - Initialize alpha and epsilon
 - Initialize Q1(s, a) and Q2(s, a)
 - Loop over episodes
  - Initialize S
  - For each episode
    - Choose A from S using epsilon greedy in Q1 + Q2
    - Take action A, observe R, S'
    - with 0.5 probability
      - Q1 -> Q1 + alpha * [R + gamma * Q2(S', argmax Q1(S', a)) - Q1]
      - Q2 -> Q2 + alpha * [R + gamma * Q1(S', argmax Q2(S', a)) - Q2]
    - S -> S'
"""


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


def max_action(Q1, Q2, state, actions=[0, 1]):
  values = np.array([Q1[state, a] + Q2[state, a] for a in actions])
  action = np.argmax(values)

  return action


def plotRunningAverage(total_rewards):
  N = len(total_rewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean(total_rewards[max(a, t-100):(t+1)])
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  n_games = 100_000
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
        

  # load = False

  # if load == False:
  Q1, Q2 = {}, {}
  for state in states:
    for action in action_space:
      Q1[state, action] = 0
      Q2[state, action] = 0
  # else:
  #   pickle_in = open('acrobot.pkl', 'rb')
  #   Q = pickle.load(pickle_in)
  #   env = wrappers.Monitor(env, "tmp/cartpole", video_callable=lambda episode_id: True, force=True)


  score = 0
  total_rewards = np.zeros(n_games)

  for i in range(n_games):
    observation = env.reset()
    done = False

    if i % 5000 == 0:
      print('episode ', i, ' score ', score, ' eps ', eps)

    score = 0
    while  not done:
      state = get_state(observation)
      action = env.action_space.sample() if np.random.random() < eps else \
                 max_action(Q1, Q2, state)

      observation_, reward, done, info = env.step(action)
      score += reward

      state_ = get_state(observation_)

      rand = np.random.random()
      if rand <= 0.5:
        action_ = max_action(Q1, Q1, state_)
        Q1[state, action] = Q1[state, action] + \
          alpha * (reward + gamma * Q2[state_, action_] - Q1[state, action])
      elif rand > 0.5:
        action_ = max_action(Q2, Q2, state_)
        Q2[state, action] = Q2[state, action] + \
          alpha * (reward + gamma * Q1[state_, action_] - Q2[state, action])

      observation = observation_

    total_rewards[i] = score
    eps = eps - 2 / n_games if eps > 0.0 else 0.0

  plotRunningAverage(total_rewards)
  # mean_rewards = np.zeros(n_games)
  # for t in range(n_games):
  #   mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])

  # figure, axis = plt.subplots(2, 1)

  # axis[0].plot(mean_rewards)
  # axis[0].set_title("mean_rewards")

  # axis[1].plot(total_rewards)
  # axis[1].set_title("total_rewards")

  # plt.show()
  # figure.savefig('rewards.png')

  # f = open(modle_name_pkl, 'wb')
  # pickle.dump(Q, f)
  # f.close()