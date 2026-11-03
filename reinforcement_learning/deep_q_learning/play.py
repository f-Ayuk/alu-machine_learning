#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""

import gym
import numpy as np
from PIL import Image

from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
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

nb_actions = env.action_space.n

# Load the trained neural network
model = load_model('policy.h5', compile=False)

# Configure memory
memory = SequentialMemory(
    limit=1000000,
    window_length=4
)

# Create the DQN agent with greedy policy
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    processor=AtariProcessor(),
    policy=GreedyQPolicy(),
    test_policy=GreedyQPolicy(),
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

# Play the game
dqn.test(
    env,
    nb_episodes=5,
    visualize=True
)

env.close()
