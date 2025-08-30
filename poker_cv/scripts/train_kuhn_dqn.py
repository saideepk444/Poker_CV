import argparse
import matplotlib.pyplot as plt
from poker_cv.rl.dqn import dqn_train, play_policy
from poker_cv.poker.envs.kuhn import KuhnPokerEnv
import torch

def make_env():
    return KuhnPokerEnv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    q, rewards = dqn_train(make_env, episodes=args.episodes)

    # Save
    torch.save(q.state_dict(), "models/kuhn_dqn.pt")
    print(f"Saved model to models/kuhn_dqn.pt")

    if args.plot:
        plt.figure()
        plt.plot(rewards)
        plt.title("Episode return (Kuhn Poker DQN)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.show()

    if args.evaluate:
        wr = play_policy(make_env, q, episodes=2000)
        print(f"Estimated win rate vs. environment baseline: {wr:.3f}")

if __name__ == "__main__":
    main()
