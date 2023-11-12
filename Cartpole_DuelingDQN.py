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

# Dueling DQN Model architecture
input_layer = Input(shape=(1, states)) # 1 is the window length
x = Flatten()(input_layer) # Flatten the input layer
x = Dense(32, activation='relu')(x) # Hidden layer
x = Dense(32, activation='relu')(x) # Hidden layer
V = Dense(1, activation='linear')(x) # Hidden layer, V is the value function
A = Dense(actions, activation='linear')(x) # Hidden layer, A is the advantage function
output_layer = Add()(
    [V, Subtract()([A, tf.reduce_mean(A, axis=1, keepdims=True)])]) # Output layer
model_dueling_dqn = Model(inputs=input_layer, outputs=output_layer)

# Dueling DQN Agent
dueling_dqn_agent = DQNAgent(model=model_dueling_dqn, memory=SequentialMemory(limit=50000, window_length=1), policy=EpsGreedyQPolicy(
), nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2, enable_dueling_network=False)
dueling_dqn_agent.compile(Adam(lr=1e-3), metrics=["mae"])

# Training the Dueling DQN Agent
dueling_dqn_train_history =  dueling_dqn_agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Plot the rewards for Dueling DQN Training
plt.plot(dueling_dqn_train_history.history['episode_reward'])
plt.title('Episode rewards for Dueling DQN Training')
plt.ylabel('Rewards')
plt.xlabel('Episode')
plt.show()

dueling_dqn_results = dueling_dqn_agent.test(
    env, nb_episodes=10, visualize=True)
print(f'Average reward for the Dueling DQN Agent: {np.mean(dueling_dqn_results.history["episode_reward"])}')

# Plot rewards for Dueling DQN Testing
plt.plot(dueling_dqn_results.history['episode_reward'], label='Dueling DQN')
plt.title('Episode rewards for Dueling DQN Testing')
plt.ylabel('Rewards')
plt.xlabel('Episode')
plt.legend()
plt.show()

env.close()
