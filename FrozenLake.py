from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

def runFrozenLake():

    # Create the FrozenLake environment
    Environment = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="8x8",
        is_slippery=False,
        #success_rate=1.0/3.0,
        render_mode=None
    )

    # Set the maximum number of steps per run to 500
    Environment = TimeLimit(Environment, max_episode_steps=500)

    # Initialize the Q-table with zeros
    q = np.zeros((Environment.observation_space.n, Environment.action_space.n))

    # Hyperparameters
    episodes = 10000
    learning_rate_alpha = 0.5
    discount_factor_gamma = 0.9
    epsilon = 1
    epsilon_decay = 0.00012

    # Initialize arrays to store the reward and number of steps for each run
    reward_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)
    total_reward_per_episode = np.zeros(episodes)
    
    
    for x in range(episodes):

        # Reset the environment for each run
        state = Environment.reset()[0]
        terminated = False
        truncated = False

        steps = 0
        total_reward = 0
        # While the episode has not reached the terminal state or been truncated because of time limit
        while (not terminated and not truncated):

            # Choose an action using the epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = Environment.action_space.sample()
            else:
                best_actions = np.flatnonzero(q[state] == np.max(q[state]))
                action = np.random.choice(best_actions)

            new_state, reward, terminated, truncated, info= Environment.step(action)
            steps += 1

            # This is to make the next max value 0 if the run ended
            if terminated or truncated:
                next_max = 0
            else:
                next_max = np.max(q[new_state])

            # Establishing a reward for the run
            if terminated:
                # If the run ended reaching the goal, we give a larger insentive
                if reward == 1:
                    reward = 10
                else:
                    # If the run ended because it fell into a hole, we give it a large negative reward
                    reward = -10
            # We assign a small cost to each step to incentivize the agent to reach the goal faster
            else:
                reward = -0.1

            total_reward += reward

            # Update the Q-value using the Q-learning formula
            q[state, action] = q[state, action] + learning_rate_alpha * (
            reward + discount_factor_gamma * next_max - q[state, action]
            )

            # Update the current state to the new state
            state = new_state
            # End of the while loop for the episode

        epsilon = max(epsilon - epsilon_decay, 0.00)

        # If the run ended reaching the goal, we record it as a success and store the number of steps it took to reach the goal
        if reward == 10:
            reward_per_episode[x] = 1
            steps_per_episode[x] = steps
        total_reward_per_episode[x] = total_reward
        # End of the training loop

    # This render is for the final run to show the best route found, and to record for the assignment
    Render = gym.make(
            'FrozenLake-v1',
            desc=None,
            map_name="8x8",
            is_slippery=False,
            render_mode="human"
        )

    state = Render.reset()[0]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        best_actions = np.flatnonzero(q[state] == np.max(q[state]))
        action = np.random.choice(best_actions)
        state, reward, terminated, truncated, info = Render.step(action)
    
    Render.close()
    # End of the render for the final run -----------------------------
    

    # This next part is to calculate the average number of steps it took to reach the goal for all successful runs
    success_steps = steps_per_episode[reward_per_episode == 1]

    print("Average steps to success: ",np.mean(success_steps))
    print("The best reward route discovered: ",total_reward_per_episode.max())
    
    # We are graphing by the success rate of the last 100 runs
    # The plot goes up to 1.0 and down to 0.0
    # We are recording the percentage of successful runs in the last 100
    success_rate = np.convolve(reward_per_episode, np.ones(100)/100, mode='valid')

    # Making the plot for total reward per episode
    plt.figure(figsize=(10, 5))
    plt.plot(total_reward_per_episode)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Making the plot for success rate
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(100, episodes+1), success_rate)
    plt.ylim(0,1.01)
    plt.xlim(0, episodes)
    plt.title("Success Rate over Episodes (Out of 100 previous attempts)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.show()

    Environment.close()
    # End of the runFrozenLake function

# Main execution
if __name__ == "__main__":
    runFrozenLake()