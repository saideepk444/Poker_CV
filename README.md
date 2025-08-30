# Computer Vision & AI for Poker

**Goal:** Build a full-stack project that (1) detects/classifies playing cards with computer vision, (2) trains RL/NN agents to play poker in a simplified environment, and (3) evaluates agents against baselines and humans.

## Project roadmap (milestones)
1) **Card CV MVP (classical OpenCV)** — Segment table, find card contours, warp cards, extract rank/suit ROIs.
2) **Card Classifier (deep learning)** — Train a small CNN with two heads (rank: 13, suit: 4) on warped ROIs.
3) **Poker RL (Kuhn → Hold’em-lite)** — Start with Kuhn Poker (tiny state/action space) via Gymnasium-style env; train DQN.
4) **Evaluation** — Play agent vs. baselines; compute win rate, decision accuracy vs. a Monte Carlo baseline, and adaptability.
5) **Stretch** — Move toward (heads-up) Hold’em-lite: discrete bet sizes, hand-strength Monte Carlo for rewards/shaping.

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

## Data collection for cards
- Put your table photos/videos in `data/raw/`. Start with top-down perspective if possible.
- Use `poker_cv/vision/detector.py` to extract warped card crops (rank/suit ROIs written to `data/processed/`).
- Label format: `processed/rank_suit/<rank>_<suit>/*.png` (e.g., `7_hearts/IMG_0012.png`).

You can generate a small synthetic set by photographing each card face under multiple angles/lighting, then augmenting (rotation, blur, minor perspective, brightness/contrast).

## Training the classifier
- The CNN in `poker_cv/vision/roi_classifier.py` is a **multi-head** model with outputs for `rank` (13) and `suit` (4).
- Use grayscale 64×64 ROIs per head; simple augmentations recommended.
- Start with ~50–200 samples per class; expand as needed.

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

## Next steps (suggested)
- [x] Run DQN on Kuhn (sanity check).
- [ ] Shoot 10–20 photos with scattered cards; run detector to extract crops.
- [ ] Manually sort a small seed set per class; train the ROI classifier and inspect misclassifications.
- [ ] Prototype **Hold’em-lite** env and plug classifier outputs into the state (your hole cards + community cards).
- [ ] Build the evaluation report (matplotlib plots) comparing policies and win rates.
