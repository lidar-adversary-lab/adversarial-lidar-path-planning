# Note: To fully reproduce the paper, you need:
# - A* for dataset generation
# - TRADES adversarial training loop
# - Multiple seeds for ensemble members
# This code provides the core components for the novel black-box attack (MI-FGSM) and defense (EDD + A* fallback).


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# -----------------------------
# Environment (20x20 Grid World)
# -----------------------------
class GridWorldEnv:
    def __init__(self, grid_size=20, obstacle_density=0.20, dynamic_prob=0.05):
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.dynamic_prob = dynamic_prob
        self.start = np.array([2, 2])
        self.goal = np.array([18, 18])
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        # Static obstacles
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        obs_indices = np.random.choice(self.grid_size * self.grid_size, num_obstacles, replace=False)
        self.grid.flat[obs_indices] = 1
        # Clear start and goal
        self.grid[self.start[1], self.start[0]] = 0
        self.grid[self.goal[1], self.goal[0]] = 0
        self.robot_pos = self.start.copy()
        self.heading = 0  # 0: North, 1: East, 2: South, 3: West
        return self.get_state()

    def step(self, action):
        # action: 0=forward, 1=turn left, 2=turn right
        if action == 1:
            self.heading = (self.heading - 1) % 4
        elif action == 2:
            self.heading = (self.heading + 1) % 4
        elif action == 0:
            dx = [0, 1, 0, -1][self.heading]
            dy = [-1, 0, 1, 0][self.heading]
            new_pos = self.robot_pos + np.array([dx, dy])
            if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size and
                self.grid[new_pos[1], new_pos[0]] == 0):
                self.robot_pos = new_pos
            else:
                return self.get_state(), -1  # collision penalty (optional)
        # Dynamic obstacles (every 5 steps, simplified here per step for demo)
        if random.random() < self.dynamic_prob:
            obs = np.argwhere(self.grid == 1)
            if len(obs) > 0:
                idx = random.choice(range(len(obs)))
                y, x = obs[idx]
                directions = [(-1,0),(1,0),(0,-1),(0,1)]
                random.shuffle(directions)
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.grid_size and 0 <= nx < self.grid_size and
                        self.grid[ny, nx] == 0 and not (nx == self.goal[0] and ny == self.goal[1])):
                        self.grid[ny, nx] = 1
                        self.grid[y, x] = 0
                        break
        done = np.all(self.robot_pos == self.goal)
        return self.get_state(), 0 if done else 0, done

    def get_lidar(self):
        distances = []
        angles = np.deg2rad(np.linspace(0, 315, 8))
        max_range = 10
        for angle in angles:
            direction = np.array([np.cos(angle + self.heading * np.pi / 2), np.sin(angle + self.heading * np.pi / 2)])
            dist = 0
            while dist < max_range:
                check_pos = self.robot_pos + direction * (dist + 1)
                x, y = int(check_pos[0]), int(check_pos[1])
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size) or self.grid[y, x] == 1:
                    break
                dist += 1
            distances.append(dist / max_range)  # normalized [0,1]
        return np.array(distances, dtype=np.float32)

    def get_state(self):
        lidar = self.get_lidar()
        heading_onehot = np.zeros(4)
        heading_onehot[self.heading] = 1
        goal_dir = self.goal - self.robot_pos
        goal_norm = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
        return np.concatenate([lidar, heading_onehot, goal_norm])

# -----------------------------
# LSTM Planner (single member)
# -----------------------------
class LSTMPlanner(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=64, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim // 5, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq):
        # seq: (batch, 5, 14) -> reshape to (batch, 5, input_per_step=14)
        out, (h, c) = self.lstm(seq)
        out = self.dropout(h[-1])
        return self.fc(out)

# -----------------------------
# Ensemble with Disagreement Detection
# -----------------------------
class EnsembleLSTMPlanner(nn.Module):
    def __init__(self, num_members=3, input_dim=70, hidden_dim=64, output_dim=3):
        super().__init__()
        self.members = nn.ModuleList([LSTMPlanner(input_dim, hidden_dim, output_dim) for _ in range(num_members)])

    def forward(self, seq):
        logits_list = [member(seq) for member in self.members]
        avg_logits = torch.stack(logits_list).mean(0)
        return avg_logits, logits_list

    def detect_adversarial(self, logits_list, threshold=0.5):
        probs = [torch.softmax(logits, dim=-1) for logits in logits_list]
        probs_stack = torch.stack(probs)  # (ensemble, batch, classes)
        std = torch.std(probs_stack, dim=0).mean(dim=-1)  # mean std per sample
        return std > threshold

# -----------------------------
# MI-FGSM Attack (transferable black-box)
# -----------------------------
def mi_fgsm_attack(model, x_seq, y, epsilon=0.30, alpha=0.03, iters=10, decay=1.0):
    """
    x_seq: tensor (1, seq_len=5, feature_dim=14)
    y: target label (scalar tensor)
    Only perturbs first 8 dims (LiDAR) across all timesteps.
    """
    x_adv = x_seq.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(x_adv)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        logits = model(x_adv)
        loss = criterion(logits, y.unsqueeze(0))
        loss.backward()

        grad = x_adv.grad
        # Mask non-LiDAR features (last 6 dims)
        mask = torch.cat([torch.ones(8), torch.zeros(6)]).to(grad.device)
        grad = grad * mask.view(1, 1, -1)

        momentum = decay * momentum + grad / (torch.norm(grad, p=1, dim=-1, keepdim=True) + 1e-10)
        x_adv = x_adv + alpha * momentum.sign()
        x_adv = torch.min(torch.max(x_adv, x_seq - epsilon), x_seq + epsilon)
        # Clip LiDAR to [0,1]
        x_adv[..., :8] = torch.clamp(x_adv[..., :8], 0.0, 1.0)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()

# -----------------------------
# Example Usage / Rollout with EDD
# -----------------------------
def robust_rollout(env, ensemble_model, surrogate_model=None, attack=False, threshold=0.5, max_steps=200):
    history = deque(maxlen=5)
    state = env.get_state()
    history.append(state)
    while len(history) < 5:
        history.append(state)

    success = False
    for step in range(max_steps):
        seq = torch.tensor(np.stack(history), dtype=torch.float32).unsqueeze(0)  # (1,5,14)

        if attack and surrogate_model is not None:
            # Use stored expert action as target (simulated)
            expert_action = 0  # placeholder; in practice retrieve from A* demo
            y = torch.tensor([expert_action])
            seq_adv = mi_fgsm_attack(surrogate_model, seq, y)
        else:
            seq_adv = seq

        avg_logits, logits_list = ensemble_model(seq_adv)
        if ensemble_model.detect_adversarial(logits_list, threshold):
            # Fallback to A* (simplified: greedy forward if clear)
            lidar = env.get_lidar()
            if lidar[0] > 0.1:  # forward ray clear
                action = 0
            else:
                action = random.choice([1, 2])
        else:
            action = torch.argmax(avg_logits, dim=-1).item()

        next_state, reward, done = env.step(action)
        history.append(next_state)

        if done:
            success = True
            break

    return success
