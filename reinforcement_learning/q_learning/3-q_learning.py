#!/usr/bin/env python3
"""Q-learning algorithm."""

import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """Train an agent using Q-learning."""

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = 0

        for step in range(max_steps):
            # Choose an action using epsilon-greedy
            action = np.random.uniform(0, 1)

            if action < epsilon:
                action = np.random.randint(Q.shape[1])
            else:
                action = np.argmax(Q[state])

            # Take the action
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Penalize falling into a hole
            if terminated and reward == 0:
                reward = -1

            # Update the Q-table
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            episode_rewards += reward
            state = new_state

            if terminated or truncated:
                break

        total_rewards.append(episode_rewards)

        # Decay epsilon
        epsilon = max(
            min_epsilon,
            epsilon - epsilon_decay * epsilon
        )

    return Q, total_rewards
