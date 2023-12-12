# https://www.youtube.com/watch?v=P9XezMuPfLE&list=PL-9x0_FO_lglnlYextpvu39E7vWzHhtNO&index=1
# https://www.gymlibrary.dev/environments/classic_control/cart_pole/

# import numpy as np
# import matplotlib.pyplot as plt
# import gym
# from gym import wrappers
# import pickle # save models
# # http://incompleteideas.net/book/ebook/node64.html

# n_discretize = 10
# cart_position_space         = np.linspace(-0.20943951, -0.20943951, n_discretize)
# cart_velocity_space         = np.linspace(-4, 4, n_discretize)
# pole_angle_space            = np.linspace(-2.4, 2.4, n_discretize)
# pole_angular_velocity_space = np.linspace(-4, 4, n_discretize)

# modle_name_pkl = 'cartpole.pkl'

# #discritize action space
# def get_state(observation):
#   cart_x_bucket_idx, cart_xdot_bucket_idx, pole_theta_bucket_idx, pole_thetadot_bucket_idx = observation

#   # gets the index of the value
#   cart_x_bucket_idx    = int(np.digitize(cart_x_bucket_idx, cart_position_space))
#   cart_xdot_bucket_idx = int(np.digitize(cart_xdot_bucket_idx, cart_velocity_space))

#   pole_theta_bucket_idx = int(np.digitize(pole_theta_bucket_idx, pole_angle_space))
#   pole_thetadot_bucket_idx = int(np.digitize(pole_thetadot_bucket_idx, pole_angular_velocity_space))

#   return (cart_x_bucket_idx, cart_xdot_bucket_idx, pole_theta_bucket_idx, pole_thetadot_bucket_idx)

# def max_action(Q, state, actions=[0, 1]):
#   values = np.array([Q[state, a] for a in actions])
#   action = np.argmax(values)

#   return action

# if __name__ == '__main__':
#   env = gym.make('CartPole-v0')
#   n_games = 50_000

#   ALPHA = 0.1
#   GAMMA = 0.99
#   EPS = 1.0

#   load = False

#   action_space = [0 , 1]
#   states = []


#   for cx in range(n_discretize + 1):
#     for cxdot in range(n_discretize + 1):
#       for pt in range(n_discretize + 1):
#         for ptdot in range(n_discretize + 1):
#           states.append((cx, cxdot, pt, ptdot))

#   if load == False:
#     Q = {}
#     for state in states:
#       for action in action_space:
#         Q[state, action] = 0
#   else:
#     pickle_in = (modle_name_pkl, 'rb')
#     Q = pickle.load(pickle_in)


#   score = 0
#   total_reward = np.zeros(n_games)

  # for i in range(n_games):
  #   obs = env.reset()
  #   done = False

  #   if i % 1000 == 0:
  #     print('episode ', i, ' score ', score, ' eps ', eps)

  #   score = 0
  #   state = get_state(obs)
  #   action = max_action(Q, state) if np.random.random() > eps else env.action_space.sample()

  #   while  not done:
  #     obs_, reward, done, info = env.step(action)
  #     state_ = get_state(obs_)
  #     action_ = max_action(Q, state_) if np.random.random() > eps else env.action_space.sample()
  #     score += reward
  #     Q[state, action] = Q[state, action] + \
  #             alpha * (reward + gamma * Q[state_, action_] - Q[state, action])
  #     state = state_
  #     action = action_

  #   total_reward[i] = score
  #   eps = eps - 2 / n_games if eps > 0.01 else 0.01

  # plt.plot(total_reward)
  # plt.show()

  # f = open(modle_name_pkl, 'wb')
  # pickle.dump(Q, f)
  # f.close()


import numpy as np
import matplotlib.pyplot as plt
import gym

def maxAction(Q, state):    
    values = np.array([Q[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

#discretize the spaces
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)

def getState(observation):
    cartX, cartXdot, cartTheta, cartThetadot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetadot = int(np.digitize(cartThetadot, poleThetaVelSpace))

    return (cartX, cartXdot, cartTheta, cartThetadot)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 0.9    
    EPS = 1.0

    #construct state space
    states = []
    for i in range(len(cartPosSpace)+1):
        for j in range(len(cartVelSpace)+1):
            for k in range(len(poleThetaSpace)+1):
                for l in range(len(poleThetaVelSpace)+1):
                    states.append((i,j,k,l))

    Q = {}
    for s in states:
        for a in range(2):
            Q[s, a] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game', i)
        # cart x position, cart velocity, pole theta, pole velocity
        observation = env.reset()        
        s = getState(observation)
        rand = np.random.random()
        a = maxAction(Q, s) if rand < (1-EPS) else env.action_space.sample()
        done = False
        epRewards = 0
        while not done:
            observation_, reward, done, info = env.step(a)   
            s_ = getState(observation_)
            rand = np.random.random()
            a_ = maxAction(Q, s_) if rand < (1-EPS) else env.action_space.sample()
            epRewards += reward
            Q[s,a] = Q[s,a] + ALPHA*(reward + GAMMA*Q[s_,a_] - Q[s,a])
            s, a = s_, a_            
        EPS -= 2/(numGames) if EPS > 0 else 0
        totalRewards[i] = epRewards

    plt.plot(totalRewards, 'b--')
    plt.show()  