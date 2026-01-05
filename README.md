# Krampus Kart: PPO Training

A browser-based reinforcement learning demo where neural network-controlled cars learn to drive around a procedurally generated track using **Proximal Policy Optimization (PPO)**.

![Krampus Kart](https://img.shields.io/badge/RL-PPO-blue) ![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-2.x-orange) ![Vanilla JS](https://img.shields.io/badge/JS-Vanilla-yellow)

## 🎯 PPO's Key Innovations

### Background: Actor-Critic
PPO uses an **actor-critic** architecture:
- **Actor** (policy network): decides what action to take given a state
- **Critic** (value network): predicts expected future reward from a state

The actor is trained using **policy gradient** — we collect experience, compute which actions were better than expected (advantages), and nudge the policy toward good actions.

### The Problem: Overtraining
With standard policy gradient, you collect a batch of experience and do **one** gradient update. If you try to reuse that same batch for multiple updates, you're essentially overtraining on stale data — the policy drifts far from the one that collected the data, making the updates invalid.

### PPO's Solution
PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) enables **multiple epochs** of updates on the same batch by clipping the objective. This prevents the policy from changing too much, keeping it "proximal" to the old policy that collected the data.

### 1. The Clipped Surrogate Objective

The goal is to align **r** (how the policy changed) with **A** (whether the action was good):
- **r** = π_new / π_old — how much more/less likely is this action now? (1.0 = unchanged)
- **A** = advantage — was this action better or worse than expected?

**The key insight:** Clipping is asymmetric.
- **When r and A agree** (making good actions more likely, or bad actions less likely): Clipping kicks in at ±10% to prevent overshooting. The direction is right, now add robustness.
- **When r and A disagree** (making good actions less likely, or bad actions more likely): No clipping. Full gradient to fix the mistake.

This lets PPO be aggressive about correcting errors, but conservative about claiming progress.

### 2. Multiple Epochs on Same Data
Unlike vanilla policy gradient (1 update per sample), PPO reuses collected experience for **K epochs** of minibatch SGD. This dramatically improves sample efficiency.

### 3. Simple First-Order Optimization  
TRPO (PPO's predecessor) required complex conjugate gradient optimization. PPO achieves similar stability with plain Adam optimizer.

---

### 📍 Where to find these in the code

The **clipped surrogate objective** is implemented in [`js/ppo/ppo-agent.js`](js/ppo/ppo-agent.js) — look for the `_updateBatch` method where it computes `ratio`, `clippedRatio`, and takes the `minimum` of the two surrogate terms.

The **hyperparameters** (clip range ε, number of epochs K, etc.) are all in [`js/config.js`](js/config.js) under the `PPO` section.

---

## 🚀 Try It Live

**[https://mattloper.github.io/krampuskart-ppo/](https://mattloper.github.io/krampuskart-ppo/)**

Or run locally:
```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

## What It Does

24 cars spawn at random positions near the start line and learn to drive through trial and error. The neural network receives sensor data and outputs steering commands (throttle is always forward). Cars are colored using a **jet colormap** based on their per-step reward (red = low, blue = high).

Over time, cars learn to maximize **forward progress** along the track.

## Architecture

Pure browser-based, no build step:
- **Simulation**: Car physics, LIDAR sensors, procedural track generation
- **PPO**: Actor-critic networks in TensorFlow.js, experience buffer, GAE
- **UI**: Canvas rendering, real-time stats

Key directories: `js/` for game logic, `js/ppo/` for the RL algorithm.

## Neural Network

**Input:** LIDAR sensor distances, speed, and angle to track direction (all normalized).

**Architecture:** Separate actor and critic networks (no shared backbone), as recommended by the PPO paper for continuous control. Small hidden layers with GELU activation.

**Output:** Continuous steering action. Throttle is always forward.

**Pretraining:** Before PPO starts, behavioral cloning teaches a simple "counter-steer proportional to angle error" policy.

## Reward Function

Reward is proportional to forward progress along the track. Simple and sparse — no shaping for clearance, alignment, or death penalties.

## PPO Hyperparameters

See [`js/config.js`](js/config.js) for all tunable parameters including:
- Discount factor (γ), GAE lambda (λ)
- Clip epsilon (ε), learning rate
- Rollout buffer size, epochs per update, batch size
- Value and entropy coefficients

## Complete Episodes Training

Only **complete episodes** (cars that crash or finish a lap) are used for training. This provides clean Monte Carlo returns without needing to bootstrap incomplete trajectories.

## Critic Training

The critic is trained using **Monte Carlo returns** (actual observed discounted rewards) rather than TD(λ) bootstrapped targets. This prevents the circular dependency where bad value predictions create bad training targets.

## Spawning & Collisions

Cars spawn at random positions near the start line. Episodes end when a car hits a wall or another car. There is no timeout.

## How Learning Works

1. **Pretrain** — Behavioral cloning teaches basic steering before PPO starts
2. **Rollout** — Cars drive around, collecting (state, action, reward, value) tuples
3. **Complete Episodes** — Only finished episodes (crashed or lapped) go to the buffer
4. **Compute Returns** — Monte Carlo returns from actual observed rewards
5. **Compute Advantages** — GAE for policy gradient
6. **PPO Update** — Clipped surrogate loss, multiple epochs on same data
7. **Repeat** — Clear buffer, keep collecting

## Dependencies

- **TensorFlow.js** (loaded from CDN)
- **Tailwind CSS** (loaded from CDN, dev only)
- Python 3 (for local server)

No npm, no build step. Just a browser and Python.

## License

MIT
