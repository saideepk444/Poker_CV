"""
Kuhn Poker DQN Training Script

This script trains a Deep Q-Network (DQN) agent to play Kuhn Poker using the DQN algorithm.
The training process involves thousands of simulated poker hands where the agent learns
optimal betting strategies through trial and error.

Training Process Overview:
1. Initialize DQN agent with random weights
2. Play multiple episodes (poker hands) against the environment
3. Store experiences in replay buffer
4. Update Q-network using sampled experiences
5. Gradually reduce exploration rate
6. Save trained model for later use

Usage Examples:
- Basic training: python train_kuhn_dqn.py
- Custom episodes: python train_kuhn_dqn.py --episodes 10000
- With plotting: python train_kuhn_dqn.py --plot
- Evaluate model: python train_kuhn_dqn.py --evaluate
- No progress bar: python train_kuhn_dqn.py --no-progress

The script automatically saves the trained model and can optionally plot training progress
or evaluate the model's performance against the environment baseline.
"""

import argparse
import matplotlib.pyplot as plt
from poker_cv.rl.dqn import dqn_train, play_policy
from poker_cv.poker.envs.kuhn import KuhnPokerEnv
import torch
import sys
import numpy as np # Added for moving average calculation

def make_env():
    """
    Factory function to create a new Kuhn Poker environment instance.
    
    This function is passed to the DQN training function and is called
    whenever a new environment is needed (e.g., for evaluation).
    
    Returns:
        KuhnPokerEnv: A fresh environment instance
    """
    return KuhnPokerEnv()

def main():
    """
    Main training function that handles command line arguments and orchestrates training.
    
    The function:
    1. Parses command line arguments
    2. Trains the DQN agent
    3. Saves the trained model
    4. Optionally plots training progress
    5. Optionally evaluates the trained model
    """
    # Set up command line argument parser with descriptive help text
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Kuhn Poker")
    
    # Training hyperparameters
    parser.add_argument("--episodes", type=int, default=5000,
                       help="Number of training episodes (default: 5000)")
    
    # Output and visualization options
    parser.add_argument("--plot", action="store_true",
                       help="Plot training progress (episode returns over time)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate trained model performance after training")
    
    # Training control options
    parser.add_argument("--log-every", type=int, default=100,
                       help="How often to log progress (default: 100 episodes)")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar display")
    
    args = parser.parse_args()

    # Main training loop with error handling for graceful interruption
    try:
        print(f"Starting DQN training for {args.episodes} episodes...")
        print(f"Logging every {args.log_every} episodes")
        print(f"Progress bar: {'enabled' if not args.no_progress else 'disabled'}")
        
        # Call the DQN training function with parsed arguments
        q, rewards = dqn_train(
            make_env,                    # Environment factory function
            episodes=args.episodes,      # Total training episodes
            log_every=args.log_every,    # Logging frequency
            progress=not args.no_progress # Whether to show progress bar
        )
        
        print(f"Training completed successfully!")
        print(f"Final model trained on {len(rewards)} episodes")
        
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C) gracefully
        print("\n[Interrupted] Training stopped by user")
        print("Saving partial model to models/kuhn_dqn_partial.pt")
        
        # Check if we have a trained model to save
        if "q" in locals():
            # Create models directory if it doesn't exist
            import os
            os.makedirs("models", exist_ok=True)
            
            # Save the partially trained model
            torch.save(q.state_dict(), "models/kuhn_dqn_partial.pt")
            print("Partial model saved successfully")
        else:
            print("No model to save - training hadn't started yet")
        
        sys.exit(130)  # Standard exit code for interrupted processes

    # Save the final trained model
    print("Saving final trained model...")
    
    # Ensure models directory exists
    import os
    os.makedirs("models", exist_ok=True)
    
    # Save the fully trained model
    torch.save(q.state_dict(), "models/kuhn_dqn.pt")
    print(f"Final model saved to models/kuhn_dqn.pt")

    # Optional: Plot training progress
    if args.plot:
        print("Generating training progress plot...")
        try:
            # Create matplotlib figure for visualization
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, alpha=0.7, linewidth=1)
            
            # Add moving average for smoother trend visualization
            window_size = min(100, len(rewards) // 10)
            if window_size > 1:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                        linewidth=2, color='red', label=f'Moving Average ({window_size})')
                plt.legend()
            
            plt.title("DQN Training Progress - Kuhn Poker")
            plt.xlabel("Episode")
            plt.ylabel("Episode Return (Chips Won/Lost)")
            plt.grid(True, alpha=0.3)
            
            # Add some statistics to the plot
            avg_reward = np.mean(rewards)
            plt.axhline(y=avg_reward, color='green', linestyle='--', 
                       label=f'Average: {avg_reward:.2f}')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"[Plot generation failed] {e}")
            print("This might be due to missing matplotlib or display issues")

    # Optional: Evaluate the trained model
    if args.evaluate:
        print("Evaluating trained model performance...")
        print("Playing 2000 hands against environment baseline...")
        
        # Evaluate the model by playing many hands and computing win rate
        wr = play_policy(make_env, q, episodes=2000)
        
        print(f"Model evaluation complete!")
        print(f"Estimated win rate vs. environment baseline: {wr:.3f} ({wr*100:.1f}%)")
        
        # Provide interpretation of the results
        if wr > 0.6:
            print("Excellent performance! The model has learned strong poker strategy.")
        elif wr > 0.55:
            print("Good performance! The model shows solid poker understanding.")
        elif wr > 0.52:
            print("Decent performance. The model has learned some basic strategy.")
        else:
            print("Performance below expectation. Consider training for more episodes or adjusting hyperparameters.")

if __name__ == "__main__":
    # Entry point: run the main function when script is executed directly
    main()
