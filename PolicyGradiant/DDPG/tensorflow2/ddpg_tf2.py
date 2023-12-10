# https://www.youtube.com/watch?v=4jh32CvwKYw&list=PL-9x0_FO_lgmP3TtVCD4X1U9oSalSuI1o&index=13
"""
DDPG uses innovations from Q learning
It uses replay memory. Instead of just learning from latest transition that the agent experienced, 
its going to keep track of the sum total of its experiences and randomly sample that memory at each
time step to get some batch of memories to update the weights of its deep neural networks.

The other innovation is use of target networks.
In Q learning we got to determine the 
1) action to take and 
2) then a network to determine the value of that action. 

The value is used to update the weights of the deep neural network.

Using the same network will make the training unstable as its chassing the tail as its a fast 
moving network as at each time step those weights are getting updated and so the evaluation 
of similar states.

Solution to this is to keep 2 networks. 
1) Value: online network to choose actions at each time 
2) Target: Evaluate the values of the action when performing the update for your deep neural network.

In this case we will be doing hard update of the target network after every 1000 steps. Its a 
hyper parameter of your agent. You would take the network parametes from the online network and copy
them to the target network. This is called the hard update.

DDPG does a soft copy of the target networks. It means we will be doing some multiplicative constant
for our update and we're going to be using a new hyper parameter called tau which is going to be a 
very small number of the order 0.001.


We will have more than one target network because DDPG is a Actor-Critic method.

Actor: What to do with what ever state we pass into it. 
Ouptputs action values rather than probabilities.

Policy - is a probability distribution of an action. What is the propabilities of any action from
the action space given a state or a set of states.


Critic : Evaluate State and Acton pairs. Given the state, the action we took we good or bad.
TargetActor
TargetCritic


deterministic - DDPG outputs action values themselfs. Pass in one state over and over again,
I will get the same action value every singe time.

This leads to explore axploit dilemma which is a fundamental problem in all RL algorithms.
The agent taking off optimial actions to explore the world is called explore-exploit dilemma.

Taking off optimial action is called exploration. Taking optimal action is called exploitation. 
As this is eploiting best known action.

Solution is to take the output of the Actor network and add some extra noise to it.
Here we are using simple gaussian noise.

Update role for actor network:
 - Randomly sample states from memory
 - Use actor to determine actions for those states
 - Plug those actions into critic and get the value
 - Take the gradient of critic network w.r.t the actor network params. This can be done
    as they are coupled based on the selections of the actions based on the actor network.

Update role for critic network:
  - Update critc by minimizing the loss. 
  This is the mean squared error between target value(yi) and Q for current state and action.
  - Randomly sample states, new states, actions, rewards from memory
  - Use target action to determine actions for new states
  - Plug those actions into target critic to get target y * gamma + reward for that time step
    which is sampled from memory. That is the target, the value you want to shift the estimates
    for the critic towards
  - Plug states, actions into critic (that is the actions the agent actually took sampled from
    memory) and take diff with target. Apply this to the MSLoss
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
  def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
          gamma=0.99, n_actions=2, max_size=1_000_000, tau=0.005,
          fc1=400, fc2=300, batch_size=64, noise=0.1):
    self.gamma = gamma
    self.tau = tau
    self.memory = ReplayBuffer(max_size, input_dims, n_actions)
    self.batch_size = batch_size
    self.n_actions = n_actions
    self.noise = noise
    self.max_action = env.action_space.high[0]
    self.min_action = env.action_space.low[0]

    self.actor = ActorNetwork(n_actions=n_actions, name='actor')
    self.critic = CriticNetwork(name='critic')
    self.target_actor = ActorNetwork(n_actions=n_actions,
                                     name='target_actor')
    self.target_critic = CriticNetwork(name='target_critic')

    self.actor.compile(optimizer=Adam(learning_rate=alpha))
    self.critic.compile(optimizer=Adam(learning_rate=beta))
    self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
    self.target_critic.compile(optimizer=Adam(learning_rate=beta))

    # copy weights from actor network to target_actor and target_critic networks
    self.update_network_parameters(tau=1)

  def update_network_parameters(self, tau=None):
    if tau is None:
      tau = self.tau

    weights = []
    targets = self.target_actor.weights

    # First iteration will be hard copy as tau will be 1
    for i, weight in enumerate(self.actor.weights):
      weights.append(weight * tau + targets[i] * (1 - tau))
    self.target_actor.set_weights(weights)

    weights = []
    targets = self.target_critic.weights
    # First iteration will be hard copy as tau will be 1
    for i, weight in enumerate(self.critic.weights):
      weights.append(weight * tau + targets[i] * (1 - tau))
    self.target_critic.set_weights(weights)

  def remember(self, state, action, reward, new_state, done):
    self.memory.store_transition(state, action, reward, new_state, done)

  def save_models(self):
    print('... saving models ...')
    self.actor.save_weights(self.actor.checkpoint_file)
    self.target_actor.save_weights(self.target_actor.checkpoint_file)
    self.critic.save_weights(self.critic.checkpoint_file)
    self.target_critic.save_weights(self.target_critic.checkpoint_file)

  def load_models(self):
    print('... loading models ...')
    self.actor.load_weights(self.actor.checkpoint_file)
    self.target_actor.load_weights(self.target_actor.checkpoint_file)
    self.critic.load_weights(self.critic.checkpoint_file)
    self.target_critic.load_weights(self.target_critic.checkpoint_file)

  def choose_action(self, observation, evaluate=False):
    state = tf.convert_to_tensor([observation], dtype=tf.float32)
    actions = self.actor(state)
    # noise for training
    if not evaluate:
      actions += tf.random.normal(shape=[self.n_actions],
              mean=0.0, stddev=self.noise)
    # clip values which are out of bounds
    actions = tf.clip_by_value(actions, self.min_action, self.max_action)

    return actions[0]

  def learn(self):
    if self.memory.mem_cntr < self.batch_size:
      return

    state, action, reward, new_state, done = \
      self.memory.sample_buffer(self.batch_size)

    states = tf.convert_to_tensor(state, dtype=tf.float32)
    states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
    actions = tf.convert_to_tensor(action, dtype=tf.float32)
    rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

    """
    Update rules for critic
    Update critic by minimizing the loss: L = 1/N SIGAMAi(yi - Q(Si, ai|theeta^Q))^2
    yi = ri + gammaQ'(si+1, mu'(si+1|theeta^mu')|theeta^Q')
    - Randomly sample states, new states, actions, rewards
    - Use target actor to determine actions for new states
    - Plug those actions into target critic to get target y
    - Plug states, actions into critic and take diff with target
    """
    
    with tf.GradientTape() as tape:
      target_actions = self.target_actor(states_)
      critic_value_ = tf.squeeze(self.target_critic(
                            states_, target_actions), 1)
      critic_value  = tf.squeeze(self.critic(states, actions), 1)
      # done is boolean flag. 1 for terminal state which will give just reward
      # none terminal state will current state value + discounted value
      target = reward + self.gamma * critic_value_ * (1 - done)
      critic_loss = keras.losses.MSE(target, critic_value)

    critic_network_gradient = tape.gradient(critic_loss,
              self.critic.trainable_variables)
    self.critic.optimizer.apply_gradients(zip(
        critic_network_gradient, self.critic.trainable_variables))


    # """
    # Update rules for actor
    # (delta theeta, mu) ~ Est ~delta^beeta[(delta theeta, mu) Q(s, a|teeta^Q) | s=st, a=mu(st|teeta,mu)]
    # mu - actor 
    # Q - critic
    # - Randomly sample state from memory
    # - Use actor to determine actions for those states
    # - Plug those actions into critic to get target y
    # - Take the gradient w.r.t the actor network params
    # """

    with tf.GradientTape() as tape:
      new_policy_actions = self.actor(states)
      actor_loss = -self.critic(states, new_policy_actions)
      actor_loss = tf.math.reduce_mean(actor_loss)

    actor_network_gradient = tape.gradient(actor_loss, 
      self.actor.trainable_variables)
    self.actor.optimizer.apply_gradients(zip(
      actor_network_gradient, self.actor.trainable_variables))
    self.update_network_parameters()