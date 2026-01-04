# Krampus Kart: PPO Training

A browser-based reinforcement learning demo where neural network-controlled cars learn to drive around a procedurally generated track using **Proximal Policy Optimization (PPO)**.

![Krampus Kart](https://img.shields.io/badge/RL-PPO-blue) ![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-2.x-orange) ![Vanilla JS](https://img.shields.io/badge/JS-Vanilla-yellow)

## Quick Start

```bash
./server.sh start
# Open http://localhost:8080/index.html
```

To stop: `./server.sh stop`

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

**Architecture:**
- 1 hidden layer, 32 units
- GELU activation
- Actor head: 1 output (steering mean) + learned log-std
- Critic head: 1 output (state value)
- Shared backbone between actor and critic

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

Cars are colored using a **jet colormap** based on their reward **this step**:
- 🔴 Red = lowest per-step reward
- 🟡 Yellow/Green = medium
- 🔵 Blue = highest per-step reward

The leader car (furthest ahead) has a white outline and visible LIDAR beams.

## Spawning & Collisions

**Spawning:**
Cars spawn at random positions near the start line (not in a fixed grid). Spawn positions are validated to be on the track.

**Episode Termination:**
Cars only die from:
- **Wall collision** - driving off the track
- **Car-car collision** - after 60-frame grace period

There is **no timeout** - cars can take as long as needed.

## Server Management

The `server.sh` script provides robust server management:

```bash
./server.sh start     # Start on port 8080 with no-cache headers
./server.sh stop      # Clean shutdown
./server.sh restart   # Stop then start
./server.sh status    # Check if running, show PID & port
```

## How Learning Works

1. **Pretrain**: Behavioral cloning teaches basic steering (20 epochs)
2. **Rollout**: 24 cars drive, collecting (state, action, reward, value) for each step
3. **Complete Episodes**: Only data from finished episodes (crashes) goes to buffer
4. **Update Trigger**: When buffer reaches 1024 samples, PPO update runs
5. **Returns**: Compute Monte Carlo returns (actual discounted rewards)
6. **Advantages**: Compute GAE advantages for policy gradient
7. **PPO Update**: 
   - Sample minibatches from buffer
   - Compute clipped surrogate loss
   - Update actor (policy) and critic (value) networks
   - Repeat for 10 epochs
8. **Repeat**: Clear buffer and continue collecting experience

## Dependencies

- **TensorFlow.js** (loaded from CDN)
- **Tailwind CSS** (loaded from CDN, dev only)
- Python 3 (for local server)

No npm, no build step. Just a browser and Python.

## License

MIT
