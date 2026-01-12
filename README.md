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

### ✅ Then immediately add the Run section
```md
### 2) Run
```bash
python robotics_adversarial_path_planning.py

---

## After that, paste your Figures + Tables
If you paste your figures/tables *right after* the Run section, they will appear.

### Quick check to avoid breaking rendering
In your `README.md`:
- Every fenced block must have **two** lines with triple backticks:
  - one to start: ```bash
  - one to end: ```
- If you forget the ending ``` line, everything below won’t render right.

If you want, paste the few lines **around** your Quick Start section from your README (like 10–15 lines), and I’ll rewrite that chunk so it’s guaranteed to render.
::contentReference[oaicite:0]{index=0}
