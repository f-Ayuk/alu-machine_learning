#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout."""

import gym
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Permute
)
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.core import Processor


class AtariProcessor(Processor):
    """Preprocess Atari observations."""

    def process_observation(self, observation):
        """Convert observations to grayscale 84x84 images."""
        img = Image.fromarray(observation)
        img = img.convert('L')
        img = img.resize((84, 84))
        return np.array(img)

    def process_state_batch(self, batch):
        """Normalize pixel values."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip rewards."""
        return np.clip(reward, -1.0, 1.0)


# Create the Breakout environment
env = gym.make('BreakoutDeterministic-v4')

# Get the number of possible actions
nb_actions = env.action_space.n

# Build the neural network
model = Sequential()
model.add(Permute(
    (2, 3, 1),
    input_shape=(4, 84, 84)
))
model.add(Conv2D(
    32, (8, 8),
    strides=(4, 4),
    activation='relu'
))
model.add(Conv2D(
    64, (4, 4),
    strides=(2, 2),
    activation='relu'
))
model.add(Conv2D(
    64, (3, 3),
    strides=(1, 1),
    activation='relu'
))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Configure experience replay
memory = SequentialMemory(
    limit=1000000,
    window_length=4
)

# Configure exploration policy
policy = EpsGreedyQPolicy()

# Create the DQN agent
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    processor=AtariProcessor(),
    policy=policy,
    nb_steps_warmup=50000,
    gamma=0.99,
    target_model_update=10000,
    train_interval=4,
    delta_clip=1.0
)

# Compile the agent
dqn.compile(
    Adam(learning_rate=0.00025),
    metrics=['mae']
)

# Train the agent
dqn.fit(
    env,
    nb_steps=1000000,
    visualize=False,
    verbose=1
)

# Save the trained policy network
dqn.model.save('policy.h5')

env.close()
