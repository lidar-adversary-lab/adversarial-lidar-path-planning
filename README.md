# Adversarial LiDAR Path Planning (Robotics)

This repository contains code and experiments for adversarial attacks and defenses on **LiDAR-based learned path planning** in a gridworld robotics simulator. We evaluate **white-box PGD** attacks, **black-box transfer** attacks, and layered defenses including **adversarial training**, **temporal modeling (LSTM)**, and a runtime defense: **Ensemble Disagreement Detection (EDD)** with **A\*** fallback.

> **Anonymity warning:** If this repo must remain anonymous, do **NOT** publish an author-identifying PDF in a **public** repository. Keep the repo private or upload an anonymized PDF.

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
> Update this section if your script filenames differ.

- `robotics_adversarial_path_planning.py` — main simulator / training / evaluation entry point  
- `ensemble_edd_transfer_attack.py` — EDD + transfer attack evaluation (if present)  
- `README.md` — project overview (this file)  
- `Robotics_Paper.pdf` — paper PDF  
- `Robotics_Paper_Pic1.PNG` … `Robotics_Paper_Pic12.PNG` — paper figures (rendered below)

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
