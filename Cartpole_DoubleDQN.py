import warnings

import random
import gym
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Subtract, Add
from tensorflow.keras.optimizers.legacy import Adam

from keras import __version__
tf.keras.__version__ = __version__

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# Establishing the ecosystem/environment
env = gym.make("CartPole-v1")
np.random.seed(123)
env.seed(123)

# Getting the number of states and actions
states = env.observation_space.shape[0]
actions = env.action_space.n

# Double DQN Model architecture
model_double_dqn = Sequential() # Sequential model
model_double_dqn.add(Flatten(input_shape=(1, states))) # Flatten the input layer
model_double_dqn.add(Dense(32, activation="relu")) # Hidden layer
model_double_dqn.add(Dense(32, activation="relu")) # Hidden layer
model_double_dqn.add(Dense(32, activation="relu")) # Hidden layer
model_double_dqn.add(Dense(actions, activation="linear")) # Output layer

# Double DQN Agent
double_dqn_agent = DQNAgent(model=model_double_dqn, memory=SequentialMemory(limit=50000, window_length=1),
                 policy=EpsGreedyQPolicy(), nb_actions=actions, nb_steps_warmup=1000, target_model_update=1e-2, enable_double_dqn=True)
double_dqn_agent.compile(Adam(lr=1e-3), metrics=["mae"])

# Training the Double DQN Agent
double_dqn_train_history = double_dqn_agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Plot the rewards for Double DQN Training
plt.plot(double_dqn_train_history.history['episode_reward'])
plt.title('Episode rewards for Double DQN Training')
plt.ylabel('Rewards')
plt.xlabel('Episode')
plt.show()

double_dqn_results = double_dqn_agent.test(env, nb_episodes=10, visualize=True)
print(f'Average reward for the Double DQN Agent: {np.mean(double_dqn_results.history["episode_reward"])}')

# Plot rewards for Double DQN Testing
plt.plot(double_dqn_results.history['episode_reward'], label='Double DQN')
plt.title('Episode rewards for Double DQN Testing')
plt.ylabel('Rewards')
plt.xlabel('Episode')
plt.legend()
plt.show()

env.close()
