import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import pickle # save models
# http://incompleteideas.net/book/ebook/node64.html

n_discretize = 10
theta_space = np.linspace(-1, 1, n_discretize)
theta_dot_space = np.linspace(-10, 10, n_discretize)
modle_name_pkl = 'acrobot.pkl'

#discritize action space
def get_state(observation):
  cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = \
    observation

  # gets the index of the value
  c_th1 = int(np.digitize(cos_theta1, theta_space))
  s_th1 = int(np.digitize(sin_theta1, theta_space))

  c_th2 = int(np.digitize(cos_theta2, theta_space))
  s_th2 = int(np.digitize(sin_theta2, theta_space))

  th1_dot = int(np.digitize(theta1_dot, theta_dot_space))
  th2_dot = int(np.digitize(theta2_dot, theta_dot_space))


  return (c_th1, s_th1, c_th2, s_th2, th1_dot, th2_dot)


def max_action(Q, state, actions=[0, 1, 2]):
  values = np.array([Q[state, a] for a in actions])
  action = np.argmax(values)

  return action



if __name__ == '__main__':
  env = gym.make('Acrobot-v1')
  n_games = 500
  alpha = 0.1
  gamma = 0.99
  eps = 0

  action_space = [0, 1, 2]

  states = []
  for c1 in range(n_discretize + 1):
    for s1 in range(n_discretize + 1):
      for c2 in range(n_discretize + 1):
        for s2 in range(n_discretize + 1):
          for dot1 in range(n_discretize + 1):
            for dot2 in range(n_discretize + 1):
              states.append((c1, s1, c2, s2, dot1, dot2))

  load = True

  if load == False:
    Q = {}
    for state in states:
      for action in action_space:
        Q[state, action] = 0
  else:
    pickle_in = open('acrobot.pkl', 'rb')
    Q = pickle.load(pickle_in)
    env = wrappers.Monitor(env, "tmp/acrobot", video_callable=lambda episode_id: True, force=True)


  score = 0
  total_reward = np.zeros(n_games)

  for i in range(n_games):
    obs = env.reset()
    done = False

    if i % 1 == 0:
      print('episode ', i, ' score ', score, ' eps ', eps)

    score = 0
    state = get_state(obs)
    action = max_action(Q, state) if np.random.random() > eps else env.action_space.sample()
    # action = env.action_space.sample() if np.random.random() < eps else \
    #              max_action(Q, state)

    while  not done:
      obs_, reward, done, info = env.step(action)
      state_ = get_state(obs_)
      action_ = max_action(Q, state_)

      score += reward
      Q[state, action] = Q[state, action] + \
              alpha * (reward + gamma * Q[state_, action_] - Q[state, action])

      state = state_
      action = action_

    total_reward[i] = score
    eps = eps - 2 / n_games if eps > 0.01 else 0.01

  mean_rewards = np.zeros(n_games)
  for t in range(n_games):
    mean_rewards[t] = np.mean(total_reward[max(0, t-50):(t+1)])
  plt.plot(mean_rewards)
  plt.show()

  f = open(modle_name_pkl, 'wb')
  pickle.dump(Q, f)
  f.close()