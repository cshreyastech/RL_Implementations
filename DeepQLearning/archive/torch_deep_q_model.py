# https://www.youtube.com/watch?v=ELE2_Mftqoc 53:00

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
  def __init__(self, ALPHA):
    super(DeepQNetwork, self).__init__()
    # 1 - grayscale
    self.conv1 = nn.Conv2d(1,   32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32,  64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 3)
    # self.fc1 = nn.Linear(128 * 23 * 16, 512)
    self.fc1 = nn.Linear(128 * 19 * 8, 512)
    # 6 -actions
    self.fc2 = nn.Linear(512, 6)
    # self.optimizer = optim.SGD(self.parameters(), lr=self.ALPHA, momentum=0.9)
    self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  # observation - sequence of frames, grey scale and 4 frames
  def forward(self, observation):
    # print("observation", len(observation))
    observation = T.Tensor(observation).to(self.device)
    # observation = T.Tensor(np.array(observation), dtype=T.uint8).to(self.device)

    # observation = observation.view(-1, 3, 210, 160).to(self.device)
    # openai gym image height x width x channels. But conv1 expects channel to come first.
    # reshape the array to accomodate that (1 channel, 185, 95 intch)
    observation = observation.view(-1, 1, 185, 95) 
    observation = F.relu(self.conv1(observation))
    observation = F.relu(self.conv2(observation))
    observation = F.relu(self.conv3(observation))
    # observation = observation.view(-1. 128 * 23 * 16).to(self.device)
    # -1 - number of frames
    observation = observation.view(-1, 128 * 19 * 8)
    observation = F.relu(self.fc1(observation))
    actions = self.fc2(observation)
    # Q value for each of the actions
    return actions

class Agent(object):
  def __init__(self, gamma, epsilon, alpha,
                maxMemorySize, epsEnd=0.05,
                replace=10000, actionSpace=[0, 1, 2, 3, 4, 5]):
    if T.cuda.is_available():
      print("cuda")
    else:
      print("cpu")

    self.GAMMA = gamma
    self.EPSILON = epsilon
    self.EPS_END = epsEnd
    self.ALPHA = alpha
    self.actionSpace = actionSpace
    self.memSize = maxMemorySize
    self.steps = 0
    self.learn_step_counter = 0
    self.memory = []
    self.memCntr = 0
    self.replace_target_cnt = replace
    # Agents estimate of current states
    self.Q_eval = DeepQNetwork(alpha)
    # Agents estimate of successive states
    self.Q_next = DeepQNetwork(alpha)

  def storeTransition(self, state, action, reward, state_):
    if self.memCntr < self.memSize:
      self.memory.append([state, action, reward, state_])
    else:
      self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
    self.memCntr += 1

  def chooseAction(self, observation):
    rand = np.random.random()
    actions = self.Q_eval.forward(observation)
    if rand < 1 - self.EPSILON:
      action = T.argmax(actions[1]).item()
    else:
      action = np.random.choice(self.actionSpace)
    self.steps += 1
    return action

  def learn(self, batch_size):
    self.Q_eval.optimizer.zero_grad()
    if self.replace_target_cnt is not None and \
      self.learn_step_counter % self.replace_target_cnt == 0:
        self.Q_next.load_state_dict(self.Q_eval.state_dict())

    if self.memCntr + batch_size < self.memSize:
      memStart = int(np.random.choice(range(self.memCntr)))
    else:
      memStart = int(np.random.choice(range(self.memSize - batch_size - 1)))

    miniBatch = self.memory[memStart : memStart + batch_size]
    memory = np.array(miniBatch, dtype=object)

    # evalute and find value of current and next state
    # convert to list because memory is an array of np objets
    Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
    Qnext = self.Q_eval.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

    # batch_size x actions
    maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
    rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
    # Qpred - Value of current set of states
    # Qtarget - Max action for next successive state
    Qtarget = Qpred
    Qtarget[:, maxA] = rewards + self.GAMMA * T.max(Qnext[1])

    # Qtarget = Qpred.clone()
    # indices = np.arrange(batch_size)
    # Qtarget[indices, maxA] = rewards + self.GAMMA * T.max(Qnext[1])

    if self.steps > 500:
      if self.EPSILON - 1e-4 > self.EPS_END:
        self.EPSILON -= 1e-4
      else:
        self.EPSILON = self.EPS_END

    # Opred.requires_grad_()
    loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
    loss.backward()
    self.Q_eval.optimizer.step()
    self.learn_step_counter += 1