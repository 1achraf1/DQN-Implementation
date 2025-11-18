import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config 
from agent import DQN 

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train():
    print("--- Starting Training ---")
    env = gym.make(config.ENV_NAME)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize Agent using config parameters
    agent = DQN(
        state_size, 
        action_size, 
        config.BUFFER_SIZE, 
        config.BATCH_SIZE,
        device, 
        config.LEARNING_RATE, 
        config.GAMMA
    )

    rewards_history = []
    epsilon = config.EPSILON_START

    for episode in range(config.NUM_EPISODES):
        state, info = env.reset()
        episode_reward = 0

        for step in range(config.MAX_STEPS):
            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.train_step()

            if done:
                break

        # Epsilon Decay
        epsilon = max(config.EPSILON_END, epsilon * config.EPSILON_DECAY)
        rewards_history.append(episode_reward)

        # Update Target Network
        if episode % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Logging
        avg_reward = np.mean(rewards_history[-100:])
        if episode % 20 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.1f} | "
                  f"Avg (100): {avg_reward:.1f} | Epsilon: {epsilon:.3f}")

        
        if avg_reward >= 475:
            print(f"\nSolved in {episode} episodes!")
            torch.save(agent.policy_network.state_dict(), config.MODEL_FILE)
            print(f"Model saved to {config.MODEL_FILE}")
            break
    else:
        print("\nMax episodes reached.")
        torch.save(agent.policy_network.state_dict(), config.MODEL_FILE)

    env.close()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(config.PLOT_FILE)
    print(f"Training plot saved as {config.PLOT_FILE}")


def save_video_of_model():
    print("\n--- Recording Video ---")
    if not os.path.exists(config.MODEL_FILE):
        print(f"Error: {config.MODEL_FILE} not found. Train the model first.")
        return

    env = gym.make(config.ENV_NAME, render_mode="rgb_array")

    env = RecordVideo(
        env, 
        video_folder=config.VIDEO_FOLDER,
        episode_trigger=lambda x: True,
        name_prefix=config.VIDEO_PREFIX
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQN(
        state_size, 
        action_size, 
        config.BUFFER_SIZE, 
        config.BATCH_SIZE,
        device, 
        config.LEARNING_RATE, 
        config.GAMMA
    )

    agent.policy_network.load_state_dict(torch.load(config.MODEL_FILE, map_location=device))
    agent.policy_network.eval()

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon=0.0) # No exploration during test
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Video recording complete. Saved to folder: '{config.VIDEO_FOLDER}'")
    print(f"Test Episode Reward: {total_reward}")

if __name__ == "__main__":
    train()
    save_video_of_model()
