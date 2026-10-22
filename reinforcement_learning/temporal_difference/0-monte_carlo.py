#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm."""
    for _ in range(episodes):
        episode = []
        state = env.reset()

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            result = env.step(action)

            # Support both old and new Gym APIs
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            episode.append((state, reward))
            state = next_state

            if done:
                break

        # Update values using first-visit Monte Carlo
        visited = set()
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = gamma * G + reward

            if state not in visited:
                V[state] += alpha * (G - V[state])
                visited.add(state)

    return V
