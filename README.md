# Adversarial LiDAR Path Planning (Robotics)

This repository contains code and experiments for adversarial attacks and defenses on **LiDAR-based learned path planning** in a gridworld robotics simulator. We evaluate **white-box PGD** attacks, **black-box transfer** attacks, and layered defenses including **adversarial training**, **temporal modeling (LSTM)**, and a runtime defense: **Ensemble Disagreement Detection (EDD)** with **A\*** fallback.

> **Anonymity warning:** If your repo must remain anonymous, do **NOT** upload a PDF that contains author names/affiliations into a **public** repository. Use a private repo or upload an anonymized PDF.

---

## Contents
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Figures (from the paper)](#figures-from-the-paper)
- [Tables (from the paper)](#tables-from-the-paper)
- [Paper](#paper)
- [License](#license)

---

## Repository Structure
> Update this section to match your actual filenames after upload.

- `robotics_adversarial_path_planning.py` — main simulator / training / evaluation entry point  
- `ensemble_edd_transfer_attack.py` — EDD + transfer-attack evaluation (if present)  
- `assets/figures/` — figures shown in this README  
- `paper/` — (optional) the paper PDF (only if anonymity allows)

---

## Quick Start

### 1) Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
