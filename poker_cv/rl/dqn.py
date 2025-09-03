"""
Deep Q-Network (DQN) Implementation for Reinforcement Learning

This module implements DQN, a value-based RL algorithm that combines Q-learning with deep neural networks.
DQN learns to approximate the optimal Q-function Q(s,a) which represents the expected future reward
for taking action 'a' in state 's' and following the optimal policy thereafter.

Key Concepts:
- Q-Learning: Learns Q-values using the Bellman equation: Q(s,a) = r + γ * max Q(s',a')
- Experience Replay: Stores and samples past experiences to break temporal correlations
- Target Network: Separate network updated less frequently to prevent moving target problem
- ε-Greedy Exploration: Balances exploration (random actions) vs exploitation (best actions)

Mathematical Foundation:
- Loss Function: L = MSE(Q_pred(s,a), r + γ * max Q_target(s',a'))
- Q-Value Update: Q(s,a) ← Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
- Exploration Decay: ε ← max(ε_min, ε * ε_decay)
"""

import random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Keep PyTorch from grabbing a ton of CPU threads on macOS
torch.set_num_threads(1)

class QNet(nn.Module):
    """
    Neural Network for approximating Q-values.
    
    Architecture: 3-layer feedforward network
    - Input: State representation (e.g., card_id, history for Kuhn Poker)
    - Hidden: 64 neurons with ReLU activation
    - Output: Q-values for each possible action
    
    The network learns to map states to action-value estimates.
    """
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),      # Input → Hidden layer 1
            nn.Linear(64, 64), nn.ReLU(),           # Hidden layer 1 → Hidden layer 2
            nn.Linear(64, n_actions)                # Hidden layer 2 → Output (Q-values)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Purpose: Store past experiences (s, a, r, s', done) and sample random batches.
    Benefits:
    1. Breaks temporal correlations that can cause training instability
    2. Allows learning from past experiences multiple times
    3. Provides diverse training data for better generalization
    
    Implementation uses a deque with fixed maximum capacity.
    """
    def __init__(self, capacity: int = 10000):
        self.buf = deque(maxlen=capacity)  # Fixed-size buffer, oldest experiences are dropped
    
    def push(self, s, a, r, s2, d):
        """
        Store a transition in the replay buffer.
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            s2: Next state
            d: Done flag (episode ended)
        """
        self.buf.append((s, a, r, s2, d))
    
    def sample(self, batch_size: int):
        """
        Sample a random batch of experiences for training.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, done_flags)
            All converted to numpy arrays for batch processing
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32), 
                np.array(s2), np.array(d, dtype=np.float32))
    
    def __len__(self):
        return len(self.buf)

def to_tensor(x):
    """Convert numpy array to PyTorch tensor with float32 dtype."""
    return torch.as_tensor(x, dtype=torch.float32)

def dqn_train(
    env_maker,
    episodes=5000,
    gamma=0.99,           # Discount factor: how much to value future rewards vs immediate
    lr=1e-3,              # Learning rate for Adam optimizer
    batch_size=64,        # Number of experiences to sample for each training step
    eps_start=1.0,        # Initial exploration rate (100% random actions)
    eps_end=0.05,         # Minimum exploration rate (5% random actions)
    eps_decay=0.995,      # Exploration decay per episode
    target_sync=200,      # How often to sync target network (steps)
    log_every=100,        # How often to log progress
    progress=True         # Whether to show progress bar
):
    """
    Train a DQN agent using the specified hyperparameters.
    
    Training Process:
    1. Initialize Q-network and target network
    2. For each episode:
       - Reset environment and get initial state
       - For each step:
         - Choose action using ε-greedy policy
         - Take action, observe reward and next state
         - Store experience in replay buffer
         - Sample batch and update Q-network
         - Periodically sync target network
       - Decay exploration rate
    
    Returns:
        q: Trained Q-network
        total_rewards: List of episode returns for analysis
    """
    try:
        from tqdm import trange
        bar = trange(episodes, disable=not progress, leave=True)
    except Exception:
        bar = range(episodes)

    # Initialize environment and get action space info
    env = env_maker()
    act_space = env.action_space
    obs_dim = 2  # (card_id, history) for Kuhn Poker

    # Initialize Q-networks: main network for training, target network for stable targets
    q = QNet(obs_dim, act_space.n)           # Main Q-network (updated every step)
    q_target = QNet(obs_dim, act_space.n)   # Target network (updated every target_sync steps)
    q_target.load_state_dict(q.state_dict()) # Start with same weights
    
    # Initialize optimizer and replay buffer
    opt = optim.Adam(q.parameters(), lr=lr)  # Adam optimizer for gradient updates
    buf = ReplayBuffer(20000)                # Large buffer for diverse experiences

    # Training state variables
    eps = eps_start                          # Current exploration rate
    total_rewards = []                       # Track performance over time
    recent = deque(maxlen=log_every)        # Moving average for recent episodes
    steps = 0                               # Total environment steps
    t0 = time.time()                        # Training start time

    for ep in bar:
        # Start new episode
        obs, _ = env.reset()
        obs = obs.astype(np.float32)        # Ensure float32 for neural network
        done = False
        ep_reward = 0.0

        while not done:
            steps += 1
            
            # ε-Greedy Action Selection
            # With probability ε: random action (exploration)
            # With probability 1-ε: best action according to Q-network (exploitation)
            if random.random() < eps:
                a = act_space.sample()       # Random action for exploration
            else:
                with torch.no_grad():        # No gradients needed for inference
                    qvals = q(to_tensor(obs).unsqueeze(0))  # Get Q-values for current state
                    a = int(qvals.argmax(dim=1).item())     # Choose action with highest Q-value

            # Take action in environment
            obs2, r, done, trunc, info = env.step(a)
            obs2 = obs2.astype(np.float32)
            
            # Store experience in replay buffer for later learning
            buf.push(obs, a, r, obs2, done)
            obs = obs2                        # Move to next state
            ep_reward += r                    # Accumulate episode reward

            # DQN Learning Step
            # Only learn if we have enough experiences in the buffer
            if len(buf) >= batch_size:
                # Sample random batch of experiences
                s, a_b, r_b, s2, d = buf.sample(batch_size)
                
                # Convert to tensors for neural network processing
                s_t = to_tensor(s)           # Current states
                s2_t = to_tensor(s2)         # Next states
                r_t = to_tensor(r_b).unsqueeze(1)  # Rewards (add batch dimension)
                a_t = torch.as_tensor(a_b, dtype=torch.int64).unsqueeze(1)  # Actions
                d_t = to_tensor(d).unsqueeze(1)    # Done flags

                # Compute current Q-values: Q(s,a) for the actions that were actually taken
                q_pred = q(s_t).gather(1, a_t)  # Select Q-values for the actions taken
                
                # Compute target Q-values using Bellman equation: r + γ * max Q(s',a')
                with torch.no_grad():  # No gradients for target computation
                    q_next = q_target(s2_t).max(dim=1, keepdim=True)[0]  # max Q(s',a')
                    q_tgt = r_t + gamma * (1 - d_t) * q_next  # r + γ * max Q(s',a') if not done

                # Compute loss: difference between predicted and target Q-values
                # This is the core of DQN learning
                loss = torch.nn.functional.mse_loss(q_pred, q_tgt)
                
                # Backpropagation and weight update
                opt.zero_grad()    # Clear previous gradients
                loss.backward()    # Compute gradients
                opt.step()         # Update weights

            # Target Network Synchronization
            # Update target network every target_sync steps to prevent moving target problem
            if steps % target_sync == 0:
                q_target.load_state_dict(q.state_dict())

        # Episode completed - update exploration rate and logging
        eps = max(eps_end, eps * eps_decay)  # Decay exploration rate, but not below eps_end
        total_rewards.append(ep_reward)
        recent.append(ep_reward)

        # Update progress bar / periodic logs
        if hasattr(bar, "set_postfix"):
            bar.set_postfix({
                "eps": f"{eps:.3f}",                    # Current exploration rate
                "avg{log}".format(log=log_every): f"{(sum(recent)/max(1,len(recent))):+.3f}",  # Recent average reward
                "steps": steps                           # Total environment steps
            })
        elif (ep + 1) % log_every == 0:
            avg_recent = sum(recent)/len(recent)
            dt = time.time() - t0
            print(f"[ep {ep+1}/{episodes}] avg_reward({log_every})={avg_recent:+.3f} eps={eps:.3f} steps={steps} time={dt:.1f}s")

    return q, total_rewards

def play_policy(env_maker, qnet, episodes=1000, eps=0.0):
    """
    Evaluate a trained Q-network by playing episodes against the environment baseline.
    
    This function tests how well the learned policy performs by:
    1. Running the trained Q-network for multiple episodes
    2. Computing win rate (fraction of positive-return episodes)
    3. Optionally adding small amount of exploration (eps > 0)
    
    Args:
        env_maker: Function that creates a new environment instance
        qnet: Trained Q-network to evaluate
        episodes: Number of episodes to play
        eps: Exploration rate (0.0 = pure exploitation, use learned policy)
    
    Returns:
        float: Win rate (0.0 to 1.0) - fraction of episodes with positive return
    """
    import numpy as np
    import torch

    env = env_maker()
    act_space = env.action_space
    wins = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        cum = 0.0  # Cumulative reward for this episode
        
        while not done:
            # Action selection with optional exploration
            if np.random.rand() < eps:
                a = act_space.sample()  # Random action
            else:
                with torch.no_grad():
                    qvals = qnet(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    a = int(qvals.argmax(dim=1).item())  # Best action according to Q-network
            
            # Take action and observe result
            obs, r, done, trunc, info = env.step(a)
            cum += r  # Accumulate reward
        
        # Count this as a win if episode had positive return
        if cum > 0:
            wins += 1
    
    return wins / episodes  # Return win rate as fraction
