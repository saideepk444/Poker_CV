# Computer Vision & AI for Poker



## Quick start
```bash
# 1) Create env and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Train a tiny DQN on Kuhn Poker (10–30s on CPU)
python -m poker_cv.scripts.train_kuhn_dqn --episodes 5000 --plot
```

This will create a checkpoint in `models/kuhn_dqn.pt` and print training stats. You can then benchmark:
```bash
python -m poker_cv.scripts.train_kuhn_dqn --evaluate --episodes 2000
```
## RL environments
- **Kuhn Poker**: 3-card deck (J,Q,K), one betting round; perfect to validate RL code quickly.
- **Hold’em-lite (stretch)**: Heads-up, discrete bet sizes (e.g., [check/call, half-pot, pot, all-in]), and limited stack sizes.

## Evaluation
- **Win rate** over N hands vs. random / rule-based baselines.
- **Decision accuracy** vs. a baseline policy (e.g., pick the action with highest Monte Carlo-estimated EV).
- **Adaptability**: Train vs. Baseline A, then switch opponent to Baseline B and measure performance drop and recovery.

## Repo structure
```
poker_cv_project/
├─ data/
│  ├─ raw/          # your photos/videos
│  └─ processed/    # warped cards + ROIs
├─ models/          # trained weights
├─ poker_cv/
│  ├─ vision/
│  │  ├─ detector.py        # OpenCV pipeline (contours + perspective warp)
│  │  └─ roi_classifier.py  # PyTorch multi-head CNN (rank + suit)
│  ├─ poker/
│  │  └─ envs/kuhn.py       # Gymnasium-style Kuhn Poker env
│  ├─ rl/
│  │  └─ dqn.py             # minimal DQN w/ replay buffer
│  ├─ eval/
│  │  └─ benchmark.py       # win-rate evaluation utilities
│  └─ scripts/
│     └─ train_kuhn_dqn.py  # train/eval entrypoint
└─ requirements.txt
```