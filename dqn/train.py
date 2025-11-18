import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from agent import DQN

ENV_NAME = "CartPole-v1"
NUM_EPISODES = 500
MAX_STEPS = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
LEARNING_RATE = 0.01
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 20
MODEL_FILE = "dqn_cartpole.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train():
    print("--- Starting Training ---")
    env = gym.make(ENV_NAME)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQN(state_size, action_size, BUFFER_SIZE, BATCH_SIZE,
                device, LEARNING_RATE, GAMMA)

    rewards_history = []
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):

            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.train_step()

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_history.append(episode_reward)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        avg_reward = np.mean(rewards_history[-100:])
        if episode % 20 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.1f} | "
                  f"Avg (100): {avg_reward:.1f} | Epsilon: {epsilon:.3f}")

        if avg_reward >= 475:
            print(f"\nSolved in {episode} episodes!")
            torch.save(agent.policy_network.state_dict(), MODEL_FILE)
            print(f"Model saved to {MODEL_FILE}")
            break
    else:
        print("\nMax episodes reached.")
        torch.save(agent.policy_network.state_dict(), MODEL_FILE)

    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_plot.png")
    print("Training plot saved as training_plot.png")


def save_video_of_model():
    print("\n--- Recording Video ---")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Train the model first.")
        return

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    video_folder = "video_results"
    env = RecordVideo(env, video_folder=video_folder,
                      episode_trigger=lambda x: True,
                      name_prefix="cartpole-agent")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size, BUFFER_SIZE, BATCH_SIZE,
                device, LEARNING_RATE, GAMMA)

    agent.policy_network.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    agent.policy_network.eval()

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon=0.0)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Video recording complete. Saved to folder: '{video_folder}'")
    print(f"Test Episode Reward: {total_reward}")


train()
save_video_of_model()
