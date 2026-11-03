#!/usr/bin/env python3
"""Play an episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Have a trained agent play one episode."""
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # Display the current state
        env.render()

        # Always exploit the Q-table
        action = np.argmax(Q[state])

        # Execute the action
        new_state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        state = new_state

        if terminated or truncated:
            break

    # Display the final state
    env.render()

    return total_rewards
