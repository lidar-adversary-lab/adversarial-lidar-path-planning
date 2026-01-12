## Paper
- [PDF](paper/Robotics_Paper (43).pdf)


## Summary
This project studies adversarial attacks on LiDAR-based learned path planning in a lightweight 20×20 gridworld simulator. A supervised MLP planner is trained from A* demonstrations and evaluated under white-box PGD and black-box transfer attacks. We also implement layered defenses including adversarial training, an LSTM temporal model, and a runtime Ensemble Disagreement Detection (EDD) mechanism that falls back to A* when an attack is detected.

## Simulator (Repro Setup)
- Grid: 20×20, ~20% static obstacles; dynamic option moves obstacles with 5% probability every 5 steps
- Start/goal: (2,2) → (18,18)
- LiDAR: 8 rays at 45° increments, max range 10 cells; normalized as d/10 if hit else 1.0
- Actions: forward 1 cell, turn left/right 90°
- A* baseline success: 98.4% over 500 maps (some unsolvable maps)
- Dataset: 2,000 maps → ~25,134 labeled transitions from A*; split by map ID to avoid leakage

## Models
- MLP: 14-dim input (8 LiDAR + 4 heading one-hot + 2 goal direction), 3-class action output
- LSTM extension: uses the last 5 LiDAR frames + current heading + goal

## Attacks
- PGD-10 (white-box): ε=0.30, α=0.03, ℓ∞ constraint; perturb LiDAR features only (masking heading/goal)
- Black-box transfer: MI-FGSM generated on a surrogate adversarially-trained LSTM and transferred to the target

## Key Results (20×20, 500 maps)
- Clean MLP success: 96.2% ± 2.9%
- Under PGD-10 ε=0.30, baseline MLP drops to 68.1% ± 6.4%
- Adversarial training (TRADES) recovers to 91.4% ± 3.6%
- Adv-trained LSTM reaches 94.2% ± 2.8% under PGD-10
- Safety: collisions increase heavily under attack; LSTM reduces collisions vs baseline defenses
- EDD (K=3 LSTMs + disagreement threshold τ=0.5) triggers A* fallback; AUROC=0.93 and recovers 93.1% success under transfer attacks, while reducing collision inflation
