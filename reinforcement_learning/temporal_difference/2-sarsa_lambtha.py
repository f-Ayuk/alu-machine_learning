
#!/usr/bin/env python3
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Performs SARSA(lambda)."""
    n_actions = Q.shape[1]

    for _ in range(episodes):
        eligibility = np.zeros_like(Q)
        state = env.reset()

        # Epsilon-greedy action selection
        if np.random.uniform() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            result = env.step(action)

            # Support both old and new Gym APIs
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            if done:
                delta = reward - Q[state, action]
                Q[state, action] += alpha * delta * (
                    eligibility[state, action] + 1
                )
                break

            # Choose the next action using epsilon-greedy
            if np.random.uniform() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA TD error
            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])

            # Update eligibility trace for the current state-action pair
            eligibility[state, action] += 1

            # Update Q table
            Q += alpha * delta * eligibility

            # Decay eligibility traces
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
