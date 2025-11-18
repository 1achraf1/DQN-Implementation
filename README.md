# Deep Q-Network (DQN) Implementation

A clean, educational implementation of Deep Q-Networks (DQN) using PyTorch. This repository demonstrates the core concepts of Reinforcement Learning including Experience Replay, Target Networks, and ε-greedy exploration, applied to the CartPole-v1 environment.

![CartPole Demo](media/cartpole_demo.gif)

## 📝 Features

- **PyTorch Implementation**: Built using modern PyTorch for tensor operations and neural network construction.
- **Experience Replay**: Implements a ReplayBuffer to store and sample transitions, stabilizing training by breaking correlation between consecutive samples.
- **Target Network**: Uses a separate target network to calculate TD targets, reducing the risk of divergence.
- **ε-Greedy Exploration**: Implements an annealing exploration strategy to balance exploration and exploitation.
- **Modular Design**: Code is structured into separate modules (agent, model, buffer) for clarity and reusability.

## 📂 Project Structure

```
DQN-Implementation/
├── dqn/
│   ├── agent.py           # DQNAgent class handling action selection and learning
│   ├── model.py           # Neural Network architecture (Q-Network)
│   ├── replay_buffer.py   # Experience Replay Buffer implementation
│   └── train.py           # Main training loop and environment interaction
├── media/                 # Images and GIFs of the agent's performance
├── .gitignore
├── requirements.txt       # (Optional) Python dependencies
└── README.md
```

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/1achraf1/DQN-Implementation.git
cd DQN-Implementation
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, simply install the core packages:

```bash
pip install torch gym numpy matplotlib
```

## 🏃 Usage

To train the DQN agent on the CartPole-v1 environment, run the training script:

```bash
python -m dqn.train
```

*(Note: Adjust the command depending on where your main entry point is located, e.g., `python dqn/train.py`)*

### Hyperparameters

You can typically adjust hyperparameters in the `train.py` file (or wherever the configuration is defined):

- `lr`: Learning Rate (e.g., 1e-3)
- `gamma`: Discount factor (e.g., 0.99)
- `buffer_size`: Capacity of the replay buffer
- `batch_size`: Number of samples per training step
- `epsilon_start`, `epsilon_end`, `epsilon_decay`: Exploration parameters

## 🧠 Algorithm Details

This implementation follows the standard DQN algorithm proposed by Mnih et al. (2015):

1. **Q-Network**: A neural network that approximates the Q-value function Q(s, a).
2. **Action Selection**: Selects action a using ε-greedy policy: random action with probability ε, otherwise arg max<sub>a</sub> Q(s, a).
3. **Storage**: Store transition (s, a, r, s', done) in the Replay Buffer.
4. **Sampling**: Sample a random batch of transitions from the buffer.
5. **Loss Calculation**: 
   
   L = E[(r + γ max<sub>a'</sub> Q<sub>target</sub>(s', a') - Q<sub>current</sub>(s, a))²]

6. **Optimization**: Perform a gradient descent step to minimize the loss.
7. **Target Update**: Periodically update the Target Network weights to match the Current Network.

## 📈 Results

The agent solves the CartPole-v1 environment (reaching a score of 500) within a few hundred episodes.

![Training Results](media/training_results.png)

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions for improvements or find any bugs.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙌 Acknowledgements

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Nature, 2015)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenAI Gym](https://www.gymlibrary.dev/)
