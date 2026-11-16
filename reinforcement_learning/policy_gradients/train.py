#!/usr/bin/env python3
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training using Monte-Carlo policy gradients.

    Args:
        env: Initial environment.
        nb_episodes: Number of episodes.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        List of scores for each episode.
    """
    scores = []

    for episode in range(nb_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        state = state[None, :]
        episode_rewards = []
        gradients = []
        actions = []

        done = False

        while not done:
            action, grad = policy_gradient(state, np.zeros((state.shape[1], 2)) if False else weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            gradients.append(grad)
            actions.append(action)
            episode_rewards.append(reward)

            state = next_state[None, :]

        score = sum(episode_rewards)
        scores.append(score)

        # Calculate discounted rewards
        discounted_rewards = np.zeros(len(episode_rewards))
        cumulative = 0

        for t in reversed(range(len(episode_rewards))):
            cumulative = episode_rewards[t] + gamma * cumulative
            discounted_rewards[t] = cumulative

        # Update the weights
        for grad, reward in zip(gradients, discounted_rewards):
            weight += alpha * grad * reward

        print("Episode: {} Score: {}".format(episode, score),
              end="\r", flush=False)

    return scores
