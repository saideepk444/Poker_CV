import numpy as np
from typing import Callable

def random_policy(obs):
    return np.random.randint(0, 2)

def evaluate(env_maker: Callable, policy: Callable, episodes: int = 2000):
    env = env_maker()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        cum = 0.0
        while not done:
            a = policy(obs)
            obs, r, done, trunc, info = env.step(a)
            cum += r
        returns.append(cum)
    win_rate = sum(1 for x in returns if x > 0) / len(returns)
    avg_return = float(np.mean(returns))
    return {"win_rate": win_rate, "avg_return": avg_return}
