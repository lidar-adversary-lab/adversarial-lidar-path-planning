# Adversarial LiDAR Path Planning (Robotics)

This repository contains code and experiments for adversarial attacks and defenses on **LiDAR-based learned path planning** in a gridworld robotics simulator. We evaluate **white-box PGD** attacks, **black-box transfer** attacks, and layered defenses including **adversarial training**, **temporal modeling (LSTM)**, and a runtime defense: **Ensemble Disagreement Detection (EDD)** with **A\*** fallback.

> **Anonymity warning:** If this repo must remain anonymous, do **NOT** publish an author-identifying PDF in a public repository. Keep the repo private or upload an anonymized PDF.

---

## Contents
- [Figures (from the paper)](#figures-from-the-paper)
- [Tables (from the paper)](#tables-from-the-paper)
- [Paper](#paper)

---
### Fig. 1 — A* Path Example (Gridworld)
![Fig. 1](./Robotics_Paper_Pic1.PNG)

This figure shows a representative **gridworld navigation** instance with an **A\***-computed path from start to goal around obstacles. It is used as a classical-planning reference point and as a fallback policy when defenses detect unsafe or unreliable learned behavior.

---

### Fig. 2 — Baseline Test Success Rate Across Runs (Dynamic Environment)
![Fig. 2](./Robotics_Paper_Pic2.PNG)

This figure summarizes baseline performance across multiple evaluation runs in a **dynamic environment** (i.e., changing maps/conditions). It highlights that even without attacks, performance can vary due to stochasticity and environment diversity—important context for interpreting robustness results.

---

### Fig. 3 — White-Box PGD Attack Sweep (Success vs. Epsilon)
![Fig. 3](./Robotics_Paper_Pic3.PNG)

This plot shows how task success degrades under a **Projected Gradient Descent (PGD)** adversarial attack as the perturbation budget (**ε**) increases. It illustrates the robustness cliff: small perturbations may be tolerated, but success can collapse past a threshold depending on model and setting.

---

### Fig. 4 — EDD Proactive Detection + Recovery (Trajectory Example)
![Fig. 4](./Robotics_Paper_Pic4.PNG)

This figure demonstrates **Ensemble Disagreement Detection (EDD)** during an episode. When ensemble members significantly disagree (a warning sign of adversarial influence or distribution shift), the defense triggers a **fallback** (e.g., A\*) to recover the trajectory and reduce unsafe actions.

---

### Fig. 5 — Robustness vs. Attack Strength (White-Box, Transfer, Transfer+EDD)
![Fig. 5](./Robotics_Paper_Pic5.PNG)

This figure compares performance under multiple threat settings: **white-box PGD**, **black-box transfer attacks**, and **transfer attacks with EDD enabled**. The key takeaway is that EDD + fallback can maintain higher success by detecting unreliable predictions instead of blindly executing compromised actions.

---

### Fig. 6 — Safety Impact (Average Collisions) Under Transfer Attack
![Fig. 6](./Robotics_Paper_Pic6.PNG)

This plot focuses on **safety**, reporting average collisions under a transfer-based adversarial setting. It shows that robustness is not only about reaching the goal—adversarial perturbations can sharply increase collision rates, motivating defenses that prioritize safe behavior.

---

### Fig. 7 — Collision Reduction Heatmap (EDD Hotspots Mitigated)
![Fig. 7](./Robotics_Paper_Pic7.PNG)

This heatmap visualizes where collisions tend to occur in the environment and how the defense shifts that distribution. The reduction of “hotspot” collision regions supports the claim that EDD + fallback improves safety by preventing repeated failure modes in high-risk areas.

---

### Fig. 8 — Mean Success Rate (Static vs Dynamic; Baseline vs PGD)
![Fig. 8](./Robotics_Paper_Pic8.PNG)

This figure compares **mean success rate** across environment types (static vs dynamic) under clean conditions and under PGD attack. It helps separate (1) difficulty due to environment complexity from (2) degradation due to adversarial perturbations.

---

### Fig. 9 — Mean Step Count (Static vs Dynamic; Baseline vs PGD)
![Fig. 9](./Robotics_Paper_Pic9.PNG)

This figure tracks **efficiency** via mean step count. Even when success remains high, attacks and defenses can affect how direct the paths are (e.g., detours, hesitation). Lower step counts generally indicate more efficient navigation among successful trials.

---

### Fig. 10 — Success Rate Comparison (MLP vs LSTM, 20×20 vs 30×30)
![Fig. 10](./Robotics_Paper_Pic10.PNG)

This plot compares architectures and scaling: **MLP vs LSTM** and smaller vs larger grid sizes. The key message is that temporal modeling (LSTM) can improve robustness/generalization in sequential decision-making, especially as task complexity increases.

---

### Fig. 11 — Additional Figure
![Fig. 11](./Robotics_Paper_Pic11.PNG)

This figure provides additional evidence supporting the paper’s evaluation (e.g., robustness, safety, ablations, or defense behavior). It should be interpreted alongside the main results to understand how attacks/defenses behave across conditions.

---

### Fig. 12 — Additional Figure
![Fig. 12](./Robotics_Paper_Pic12.PNG)

This figure complements the earlier plots by highlighting another aspect of the experimental findings (e.g., comparisons, distributions, or defense dynamics). Together with the other figures, it supports the broader conclusion that layered defenses can improve both success and safety.

---

## Tables (from the paper)

### Table I — Robust success under full PGD (ε=0.30) for varying LiDAR beam counts
| Beams | MLP Success (%) | LSTM Success (%) |
|------:|-----------------:|-----------------:|
| 8     | 91.4 ± 3.6       | 94.2 ± 2.8       |
| 16    | 93.8 ± 3.1       | 96.1 ± 2.4       |
| 32    | 95.2 ± 2.7       | 97.3 ± 2.1       |

### Table II — Robust success under partial-beam PGD (ε=0.30)
| Attack Type        | k=2   | k=4   | k=6   | Full (k=8) |
|-------------------|------:|------:|------:|-----------:|
| Random subset     | 96.5% | 95.1% | 94.7% | 94.2%      |
| Contiguous sector | 95.8% | 93.9% | 92.6% | 94.2%      |

### Table III — Model size and inference latency
| Model                          | Parameters | Latency (ms/step) |
|-------------------------------|-----------:|-------------------:|
| MLP (baseline)                | 9,731      | 0.12 ± 0.02        |
| LSTM (70 → 64 hidden)         | 51,843     | 0.38 ± 0.05        |
| Wide MLP (approx. 52k params) | 52,099     | 0.21 ± 0.03        |

### Table IV — Robust success under PGD (ε=0.30) with matched capacity
| Model              | Robust Success (%) |
|-------------------|-------------------:|
| MLP (no defense)  | 91.4 ± 3.6         |
| Wide MLP (≈52k)   | 92.6 ± 3.3         |
| LSTM (≈52k)       | 94.2 ± 2.8         |

### Table V — Robust accuracy and efficiency under strong attack
| Model               | Success (%)   | Steps (successful trials) |
|--------------------|--------------:|---------------------------:|
| MLP (no defense)   | 68.1 ± 6.4    | 94 ± 29                    |
| MLP (adv-trained)  | 91.4 ± 3.6    | 72 ± 15                    |
| LSTM (adv-trained) | 94.2 ± 2.8    | 69 ± 14                    |

### Table VI — Average collisions per trial under strongest attack (dynamic env., 500 maps)
| Condition         | Avg. Collisions per Trial |
|------------------|---------------------------:|
| Clean            | 1.8 ± 1.4                  |
| No defense       | 8.7 ± 4.9                  |
| MLP adv-trained  | 3.2 ± 2.1                  |
| LSTM adv-trained | 2.4 ± 1.7                  |

### Table VII — Robust success under PGD for varying LiDAR range (adv-trained, dynamic env., 500 maps)
| Max Range | ε (norm.) | LSTM Success (%) |
|----------:|----------:|-----------------:|
| 10 cells  | 0.30      | 94.2 ± 2.8       |
| 20 cells  | 0.30      | 91.3 ± 3.5       |
| 20 cells  | 0.15      | 95.8 ± 2.9       |

### Table VIII — Collisions per trial under transfer attack (dynamic env., 500 maps)
| Model                  | Avg. Collisions |
|-----------------------|----------------:|
| Adv. LSTM (transfer)  | 5.4 ± 3.2       |
| Ensemble Averaging    | 4.1 ± 2.4       |
| EDD + A* Fallback     | 2.6 ± 1.8       |

### Table IX — Distribution of collisions over trial timeline (% of total in each quartile)
| Condition         | 0–25% | 25–50% | 50–75% | 75–100% |
|------------------|------:|-------:|-------:|--------:|
| Clean (dynamic)  | 28%   | 32%    | 25%    | 15%     |
| No defense       | 12%   | 22%    | 38%    | 28%     |
| MLP adv-trained  | 24%   | 30%    | 29%    | 17%     |
| LSTM adv-trained | 27%   | 33%    | 26%    | 14%     |

---

## Paper
- [Robotics_Paper.pdf](./Robotics_Paper.pdf)
