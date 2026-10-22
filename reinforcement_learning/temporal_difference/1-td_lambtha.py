#!/usr/bin/env python3
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm."""
    for _ in range(episodes):
        eligibility = np.zeros_like(V)
        state = env.reset()

        for _ in range(max_steps):
            action = policy(state)
            result = env.step(action)

            # Support both old and new Gym APIs
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # TD error
            next_value = 0 if done else V[next_state]
            delta = reward + gamma * next_value - V[state]

            # Update eligibility trace
            eligibility[state] += 1

            # Update value estimates
            V += alpha * delta * eligibility

            # Decay eligibility traces
            eligibility *= gamma * lambtha

            state = next_state
            if done:
                break

    return V
