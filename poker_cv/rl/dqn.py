import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(s2), np.array(d, dtype=np.float32))
    def __len__(self):
        return len(self.buf)

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)

def dqn_train(env_maker, episodes=5000, gamma=0.99, lr=1e-3, batch_size=64, eps_start=1.0, eps_end=0.05, eps_decay=0.995, target_sync=200):
    env = env_maker()
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = 2  # (card_id, history) for Kuhn

    q = QNet(obs_dim, act_space.n)
    q_target = QNet(obs_dim, act_space.n)
    q_target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(20000)

    eps = eps_start
    total_rewards = []
    steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        done = False
        ep_reward = 0.0

        while not done:
            steps += 1
            if random.random() < eps:
                a = act_space.sample()
            else:
                with torch.no_grad():
                    qvals = q(to_tensor(obs).unsqueeze(0))
                    a = int(qvals.argmax(dim=1).item())

            obs2, r, done, trunc, info = env.step(a)
            obs2 = obs2.astype(np.float32)
            buf.push(obs, a, r, obs2, done)
            obs = obs2
            ep_reward += r

            # learn
            if len(buf) >= batch_size:
                s, a_b, r_b, s2, d = buf.sample(batch_size)
                s_t = to_tensor(s)
                s2_t = to_tensor(s2)
                r_t = to_tensor(r_b).unsqueeze(1)
                a_t = torch.as_tensor(a_b, dtype=torch.int64).unsqueeze(1)
                d_t = to_tensor(d).unsqueeze(1)

                q_pred = q(s_t).gather(1, a_t)
                with torch.no_grad():
                    q_next = q_target(s2_t).max(dim=1, keepdim=True)[0]
                    q_tgt = r_t + gamma * (1 - d_t) * q_next

                loss = torch.nn.functional.mse_loss(q_pred, q_tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if steps % target_sync == 0:
                q_target.load_state_dict(q.state_dict())

        eps = max(eps_end, eps * eps_decay)
        total_rewards.append(ep_reward)

    return q, total_rewards

def play_policy(env_maker, qnet, episodes=1000, eps=0.0):
    env = env_maker()
    act_space = env.action_space
    wins = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        cum = 0.0
        while not done:
            if np.random.rand() < eps:
                a = act_space.sample()
            else:
                with torch.no_grad():
                    qvals = qnet(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    a = int(qvals.argmax(dim=1).item())
            obs, r, done, trunc, info = env.step(a)
            cum += r
        if cum > 0: wins += 1
    return wins / episodes
