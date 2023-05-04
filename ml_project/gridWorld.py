import random

import numpy as np

# Define the grid world environment
grid_world = np.array(
    [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
)
grid_world[3, 3] = 10  # Goal state
grid_world[1, 1] = -5  # Obstacle state


# Define the Q-learning algorithm
def q_learning(
    grid_world, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1
):
    # Initialize Q-table with zeros
    Q = np.zeros((4, 4, 4))  # state-action value function

    # Loop over episodes
    for episode in range(num_episodes):
        # Initialize state
        state = (0, 0)
        # Loop over time steps
        while True:
            # Choose action using epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # random action
            else:
                action = np.argmax(Q[state[0], state[1], :])  # greedy action

            # Perform action and observe reward and next state
            if action == 0 and state[0] > 0:  # move up
                next_state = (state[0] - 1, state[1])
                reward = grid_world[next_state[0], next_state[1]]
            elif action == 1 and state[0] < 3:  # move down
                next_state = (state[0] + 1, state[1])
                reward = grid_world[next_state[0], next_state[1]]
            elif action == 2 and state[1] > 0:  # move left
                next_state = (state[0], state[1] - 1)
                reward = grid_world[next_state[0], next_state[1]]
            elif action == 3 and state[1] < 3:  # move right
                next_state = (state[0], state[1] + 1)
                reward = grid_world[next_state[0], next_state[1]]
            else:  # invalid action
                next_state = state
                reward = -100

            # Update Q-table
            Q[state[0], state[1], action] = (1 - learning_rate) * Q[
                state[0], state[1], action
            ] + learning_rate * (
                reward + discount_factor * np.max(Q[next_state[0], next_state[1], :])
            )

            # Update state
            state = next_state

            # Check if goal state is reached
            if state == (3, 3):
                break

    return Q


def main():
    """Run the gridWorld script"""
    # Run Q-learning and print the learned Q-table
    Q = q_learning(grid_world)
    print(Q)


if __name__ == "__main__":
    main()
