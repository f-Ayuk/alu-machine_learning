#!/usr/bin/env python3
"""Load the FrozenLake environment from OpenAI Gym."""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLake environment."""
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
