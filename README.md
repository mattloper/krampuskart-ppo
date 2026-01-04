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

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$ — how much the policy changed
- $\epsilon = 0.1$ — the clip range (keeps ratio in $[0.9, 1.1]$)
- $\hat{A}_t$ — advantage: "was this action better than expected?"

The clipping zeroes out gradients when the policy tries to change too much, keeping updates "proximal" (close) to the old policy.

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

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
├─────────────────────────────────────────────────────────────┤
│  index.html                                                  │
│  ├── TensorFlow.js (CDN)                                    │
│  ├── styles.css                                             │
│  └── js/                                                    │
│       ├── main.js          ← Game loop, PPO training        │
│       ├── car.js           ← Car physics, sensors, state    │
│       ├── track.js         ← Procedural track, SDF, progress│
│       ├── spline.js        ← Catmull-Rom spline math        │
│       ├── config.js        ← All hyperparameters            │
│       ├── ui.js            ← HUD updates                    │
│       ├── utils.js         ← Helper functions               │
│       ├── charts.js        ← Avg reward chart               │
│       ├── nn-visualizer.js ← Neural network weight viz      │
│       ├── simulation.js    ← Car spawning, camera, helpers  │
│       ├── debug-logger.js  ← Step & update logging          │
│       └── ppo/                                              │
│            ├── actor-critic.js    ← Neural network (TF.js)  │
│            ├── ppo-agent.js       ← PPO algorithm           │
│            ├── experience-buffer.js ← Rollout storage, GAE  │
│            └── reward.js          ← Reward computation      │
└─────────────────────────────────────────────────────────────┘
```

## Neural Network

**Input (10 dimensions):**
- 8 LIDAR sensor distances (normalized 0-1, max range 600)
- Current speed (normalized)
- Signed angle to track direction (normalized to [-1, 1])

**Architecture (Separate Networks):**
- **Actor Network**: Input → 4 hidden units (GELU) → steering mean + learned log-std
- **Critic Network**: Input → 4 hidden units (GELU) → state value
- Networks are **separate** (no shared backbone), as recommended by the PPO paper for continuous control

See **[`js/ppo/actor-critic.js`](js/ppo/actor-critic.js)** for implementation.

**Output (1 continuous action):**
- Steering: relative turn rate [-1, 1] (added to current heading each frame)
- Throttle is hardcoded to always forward (1.0)

**Pretraining:**
Before PPO starts, the network is pretrained with behavioral cloning for 20 epochs to learn a simple policy: counter-steer proportional to angle error.

**Visualization:**
The neural network weights are visualized in the lower-right corner using a jet colormap (red = negative weights, blue = positive weights).

## Reward Function

```
reward = deltaProgress × 500
```

The reward is simply proportional to forward progress along the track. Clearance, angle alignment, and death penalties are all disabled for simplicity.

## PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.995 | Discount factor (long horizon) |
| `GAE_LAMBDA` | 0.95 | Advantage estimation |
| `CLIP_EPSILON` | 0.1 | Policy clip range |
| `LEARNING_RATE` | 3e-4 | Adam optimizer LR |
| `ROLLOUT_LENGTH` | 1024 | Buffer size before update |
| `EPOCHS_PER_UPDATE` | 10 | PPO epochs per rollout |
| `BATCH_SIZE` | 64 | Minibatch size |
| `VALUE_COEF` | 10.0 | Critic loss weight |
| `ENTROPY_COEF` | 0.01 | Exploration bonus |

## Complete Episodes Training

Only **complete episodes** (cars that crash or finish a lap) are used for training. This provides clean Monte Carlo returns without needing to bootstrap incomplete trajectories.

## Critic Training

The critic is trained using **Monte Carlo returns** (actual observed discounted rewards) rather than TD(λ) bootstrapped targets. This prevents the circular dependency where bad value predictions create bad training targets.

## UI Indicators

| Indicator | Meaning |
|-----------|---------|
| **PPO Updates** | Number of policy updates |
| **Best Episode** | Highest episode reward seen |
| **Cars Active** | Cars currently driving (not crashed) |
| **Avg Reward** | Rolling average of recent episode rewards |
| **Leader Progress** | Best car's track progress (% of lap) |
| **Pred / Real / Err** | Critic prediction vs actual discounted return |
| **Progress Reward** | Average progress reward per episode |
| **Grads: ✓/✗** | Whether gradients are flowing |

## Visualizations

**Lower Left - Avg Reward Chart:**
Shows average episode reward over time (green line).

**Lower Right - Neural Network:**
Shows network weights as colored connections using a jet colormap:
- 🔴 Red = negative weights
- 🟢 Green/Yellow = near-zero weights  
- 🔵 Blue = positive weights

## Car Colors

Cars are colored using a **jet colormap** based on their **cumulative episode reward**:
- 🔴 Red = just spawned (low accumulated reward)
- 🟡 Yellow/Green = medium progress
- 🔵 Blue = survived longest / most reward

The leader car (furthest ahead) has a white outline and visible LIDAR beams.

## Spawning & Collisions

**Spawning:**
Cars spawn at random positions near the start line (not in a fixed grid). Spawn positions are validated to be on the track.

**Episode Termination:**
Cars only die from:
- **Wall collision** - driving off the track
- **Car-car collision** - after 60-frame grace period

There is **no timeout** - cars can take as long as needed.

## How Learning Works

| Step | What happens | Code |
|------|--------------|------|
| 1. **Pretrain** | Behavioral cloning teaches basic steering | [`actor-critic.js:pretrain()`](js/ppo/actor-critic.js) |
| 2. **Rollout** | 24 cars drive, collecting (state, action, reward, value) | [`main.js:ppoStep()`](js/main.js) |
| 3. **Complete Episodes** | Only crashed/finished cars' data goes to buffer | [`main.js:resetFinishedCars()`](js/main.js) |
| 4. **Returns** | Compute Monte Carlo returns (actual discounted rewards) | [`experience-buffer.js`](js/ppo/experience-buffer.js) |
| 5. **Advantages** | Compute GAE advantages for policy gradient | [`experience-buffer.js`](js/ppo/experience-buffer.js) |
| 6. **PPO Update** | **The key part!** Clipped surrogate loss, 10 epochs | [`ppo-agent.js:_updateBatch()`](js/ppo/ppo-agent.js) |
| 7. **Repeat** | Clear buffer, continue collecting | [`ppo-agent.js:update()`](js/ppo/ppo-agent.js) |

## Dependencies

- **TensorFlow.js** (loaded from CDN)
- **Tailwind CSS** (loaded from CDN, dev only)
- Python 3 (for local server)

No npm, no build step. Just a browser and Python.

## License

MIT
