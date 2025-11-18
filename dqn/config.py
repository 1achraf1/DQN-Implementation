# Hyperparameters and Configuration

# Environment
ENV_NAME = "CartPole-v1"

# Training Parameters
NUM_EPISODES = 500
MAX_STEPS = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
LEARNING_RATE = 0.01
GAMMA = 0.9

# Exploration (Epsilon Greedy)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Updates and Saving
TARGET_UPDATE_FREQ = 20
MODEL_FILE = "dqn_cartpole.pth"
VIDEO_FOLDER = "video_results"
VIDEO_PREFIX = "cartpole-agent"
PLOT_FILE = "training_plot.png"
