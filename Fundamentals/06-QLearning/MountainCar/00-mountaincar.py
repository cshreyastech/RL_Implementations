# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
# https://www.youtube.com/watch?v=rBzOyjywtPw&list=PL-9x0_FO_lglnlYextpvu39E7vWzHhtNO&index=6
# https://www.gymlibrary.dev/environments/classic_control/mountain_car/

"""
Q-Learning - off policy TD control for estimatin pi ~ pi*
Initialize Q(s, a) for all s E S, a E A, arbitarily and Q(terminal-state,) = 0
Repeat for each episode:
  Initialize S
  Repeat for each step of episode:
    Choose A from S using policy derived from Q(e.g, e-greedy)
    Take action A, observe R, S'
    Q(S, A) <- Q(S,A) + alpha[R + gamma*maxa * Q(S', a) - Q(S, A)]
    S <- S'
  until S is terminal

SARSA (State-Action-Reward-State-Action) and Q-learning are both reinforcement learning algorithms used to train agents in decision-making processes. Here are the key differences between SARSA and Q-learning:

    Update Rule:
        SARSA (On-Policy): SARSA is an on-policy algorithm, meaning that it learns the value of the policy being used to make decisions. The Q-value update in SARSA is based on the current policy's action selection.
        Q-learning (Off-Policy): Q-learning is an off-policy algorithm, which means it learns the value of the optimal policy regardless of the policy being followed. The Q-value update in Q-learning considers the maximum Q-value for the next state, irrespective of the action chosen by the current policy.

    Learning Strategy:
        SARSA: SARSA learns by directly estimating the Q-values for state-action pairs and updates its estimates as the agent interacts with the environment following a specific policy.
        Q-learning: Q-learning learns by estimating the Q-values for state-action pairs but updates its estimates using the maximum Q-value of the next state, even if the agent does not choose the action associated with the maximum Q-value.

    Exploration vs. Exploitation:
        SARSA: SARSA incorporates the exploration strategy of the current policy. It tends to be more conservative as it explores the environment while adhering to the current policy.
        Q-learning: Q-learning is more explorative because it updates its Q-values based on the maximum estimated future return, which may involve actions not chosen by the current policy.

    Convergence:
        SARSA: SARSA is generally more conservative and may converge to a suboptimal policy if the exploration strategy is not balanced.
        Q-learning: Q-learning is more likely to converge to the optimal policy, especially in scenarios with high exploration.

    Policy Update:
        SARSA: The policy is explicitly updated in SARSA, as it is an on-policy algorithm. The action chosen for the next step directly impacts the policy.
        Q-learning: Q-learning does not explicitly update the policy during the learning process. The policy is implicitly updated based on the learned Q-values.

    Use Cases:
        SARSA: SARSA is often preferred in scenarios where the exploration strategy needs to be consistent with the current policy, such as in online learning or applications with strict safety constraints.
        Q-learning: Q-learning is suitable for scenarios where more aggressive exploration is desirable, and the goal is to find the optimal policy.

In summary, SARSA and Q-learning differ in their approach to learning and updating Q-values, with SARSA being an on-policy algorithm that updates Q-values based on the current policy and Q-learning being an off-policy algorithm that updates Q-values based on the optimal policy.

"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import pickle # save models

n_car_postion_discretize = 12
n_car_velocity_discretize = 20
car_postion_space  = np.linspace(-1.2, -0.6, n_car_postion_discretize)
car_velocity_space = np.linspace( 0.6,   0.7, n_car_velocity_discretize)


modle_name_pkl = 'mountaincar.pkl'

#discritize action space
def get_state(observation):
  car_postion, car_velocity = observation

  # gets the index of the value
  car_postion_bin  = int(np.digitize(car_postion, car_postion_space))
  car_velocity_bin = int(np.digitize(car_velocity, car_velocity_space))

  return (car_postion_bin, car_velocity_bin)


def max_action(Q, state, actions=[0, 1, 2]):
  values = np.array([Q[state, a] for a in actions])
  action = np.argmax(values)

  return action


if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  env._max_episode_steps = 1000
  n_games = 50_000
  alpha = 0.1
  gamma = 0.99
  eps = 1.0

  action_space = [0, 1, 2]

  states = []
  for car_pos in range(n_car_velocity_discretize + 1):
    for car_vel in range(n_car_velocity_discretize + 1):
      states.append((car_pos, car_vel))
        

  load = False

  if load == False:
    Q = {}
    for state in states:
      for action in action_space:
        Q[state, action] = 0
  else:
    pickle_in = open(modle_name_pkl, 'rb')
    Q = pickle.load(pickle_in)
    env = wrappers.Monitor(env, "tmp/mountaincar", video_callable=lambda episode_id: True, force=True)


  score = 0
  total_rewards = np.zeros(n_games)

  for i in range(n_games):
    obs = env.reset()
    done = False

    if i % 1000 == 0:
      print('episode ', i, ' score ', score, ' eps %.3f' % eps)

    score = 0
    state = get_state(obs)

    while  not done:
      action = env.action_space.sample() if np.random.random() < eps else \
                   max_action(Q, state)
      obs_, reward, done, info = env.step(action)
      state_ = get_state(obs_)
      score += reward


      action_ = env.action_space.sample() if np.random.random() < eps else \
                 max_action(Q, state_)

      Q[state, action] = Q[state, action] + \
              alpha * (reward + gamma * Q[state_, action_] - Q[state, action])

      state = state_

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