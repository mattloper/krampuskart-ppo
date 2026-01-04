import { CONFIG, MODULE_ID, LOAD_TIME } from './config.js';
import { Track } from './track.js';
import { PPOAgent } from './ppo/ppo-agent.js';
import { computeReward } from './ppo/reward.js';
import { updateUI } from './ui.js';
import { debugLogger } from './debug-logger.js';
import { drawRewardChart } from './charts.js';
import { drawNNVisualization, extractNetworkInfo } from './nn-visualizer.js';
import { spawnCars, findLeader, resetFinishedCars, updateCamera } from './simulation.js';
import { average, pushWithLimit } from './utils.js';

console.log(`🚗 Krampus Kart PPO - Module ID: ${MODULE_ID} loaded at ${LOAD_TIME}`);

// Track centerline definition
const TRACK_CENTERLINE = [
    { x: 200, y: -25 },
    { x: 550, y: -25 },
    { x: 875, y: -25 },
    { x: 1150, y: 300 },
    { x: 975, y: 650 },
    { x: 1225, y: 775 },
    { x: 1300, y: 1025 },
    { x: 950, y: 1300 },
    { x: 550, y: 1250 },
    { x: 450, y: 1075 },
    { x: 300, y: 1100 },
    { x: 75, y: 1300 },
    { x: -425, y: 1125 },
    { x: -475, y: 625 },
    { x: -150, y: 600 },
    { x: -350, y: 400 },
    { x: -50, y: 100 },
];

// Game state - cleaned up to only include what's actually used
const state = {
    track: null,
    cars: [],
    agent: null,
    camera: { x: 0, y: 0 },
    
    // Training stats
    totalSteps: 0,
    updateCount: 0,
    recentRewards: [],  // Rolling window of episode rewards (max 100)
    bestEpisodeReward: -Infinity,
    
    // Reward history for charting
    rewardHistory: [],
    maxHistoryLen: 100,
    
    // Critic accuracy tracking
    criticPredictions: [],  // V₀ predictions at episode start
    actualReturns: [],      // Actual discounted returns
    avgPrediction: 0,
    avgActual: 0,
};

// Canvas setup
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const nnCanvas = document.getElementById('nnCanvas');
const nnCtx = nnCanvas.getContext('2d');
const lossCanvas = document.getElementById('lossChart');
const lossCtx = lossCanvas.getContext('2d');

async function setup() {
    // Update build time in UI
    setTimeout(() => {
        const buildTimeEl = document.getElementById('build-time');
        if (buildTimeEl) {
            buildTimeEl.textContent = `ID:${MODULE_ID} @ ${LOAD_TIME}`;
        }
    }, 100);
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Initialize track
    state.track = new Track(TRACK_CENTERLINE);
    
    // Initialize PPO agent
    state.agent = new PPOAgent();
    
    // Pretrain with simple heuristics
    await state.agent.pretrain();
    
    // Spawn cars at random positions near start line
    state.cars = spawnCars(state.track, CONFIG.NUM_ENVS, {
        lateralSpread: 60,
        longitudinalSpread: 200
    });
    
    requestAnimationFrame(loop);
}

function loop() {
    // Clear
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // PPO step for each car
    for (const car of state.cars) {
        if (!car.dead && !car.finished) {
            ppoStep(car);
        }
    }
    
    // Find leader for camera
    const { leader, aliveCount } = findLeader(state.cars);
    
    // Reset dead/finished cars
    resetFinishedCars(state.cars, state.track, (car) => {
        // Flush episode to training buffer
        for (const step of car.trajectory) {
            state.agent.store(step.state, step.action, step.reward, step.value, step.logProb, step.done);
        }
        state.totalSteps += car.trajectory.length;
        
        // Track critic accuracy
        if (car.episodeLength > 0) {
            pushWithLimit(state.criticPredictions, car.criticPrediction, 100);
            pushWithLimit(state.actualReturns, car.episodeDiscountedReturn, 100);
            state.avgPrediction = average(state.criticPredictions);
            state.avgActual = average(state.actualReturns);
        }
        
        // Track episode reward
        pushWithLimit(state.recentRewards, car.episodeReward, 100);
        if (car.episodeReward > state.bestEpisodeReward) {
            state.bestEpisodeReward = car.episodeReward;
        }
    });
    
    // Record critic prediction for newly spawned cars
    for (const car of state.cars) {
        if (car.episodeLength === 0 && !car.dead && !car.finished) {
            const startState = car.getStateVector(state.track);
            car.criticPrediction = state.agent.getValue(startState);
        }
    }
    
    // Update camera
    if (leader) {
        updateCamera(state.camera, leader, CONFIG.CAMERA_SMOOTHING);
    }
    
    // Draw world
    ctx.save();
    ctx.translate(canvas.width / 2 - state.camera.x, canvas.height / 2 - state.camera.y);
    
    state.track.draw(ctx);
    
    // Update car colors based on episode reward
    const aliveCars = state.cars.filter(c => !c.dead && !c.finished);
    if (aliveCars.length > 0) {
        const rewards = aliveCars.map(c => c.episodeReward || 0);
        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);
        const range = maxReward - minReward || 1;
        
        for (const car of aliveCars) {
            const normalized = (car.episodeReward - minReward) / range;
            car.setRewardColor(normalized);
        }
    }
    
    for (const car of state.cars) {
        car.draw(ctx, car === leader && !car.dead);
    }
    
    ctx.restore();
    
    // Draw charts
    const networkInfo = extractNetworkInfo(state.agent.model);
    drawNNVisualization(nnCtx, nnCanvas.width, nnCanvas.height, networkInfo);
    drawRewardChart(lossCtx, lossCanvas.width, lossCanvas.height, state.rewardHistory);
    
    // Check for PPO update
    if (state.agent.shouldUpdate()) {
        performPPOUpdate();
    }
    
    // Update UI
    const avgReward = average(state.recentRewards);
    const agentStats = state.agent.getStats();
    
    // Exploration level as percentage
    const logStd = agentStats.logStd[0] || -1;
    const noisePercent = Math.round(Math.exp(logStd) * 100);
    
    // Status message
    const statusMessage = state.agent.isUpdating 
        ? '🧠 Learning from experience...'
        : `🚗 Collecting: ${agentStats.episodeCount}/${agentStats.minEpisodes} episodes`;
    
    // Actor stats: policy loss and exploration
    const policyLoss = agentStats.lastLoss?.policy;
    const actorStats = policyLoss !== undefined 
        ? `loss ${policyLoss.toFixed(4)} | explore ${noisePercent}%`
        : `explore ${noisePercent}%`;
    
    // Critic stats: value loss and prediction accuracy
    const valueLoss = agentStats.lastLoss?.value;
    const errorPct = state.avgActual !== 0 
        ? Math.abs((state.avgPrediction - state.avgActual) / state.avgActual * 100).toFixed(0) 
        : '?';
    const criticStats = valueLoss !== undefined
        ? `loss ${valueLoss.toFixed(2)} | ${errorPct}% pred error`
        : state.avgActual > 0 ? `${errorPct}% pred error` : '—';
    
    updateUI({
        generation: state.updateCount,
        bestFitness: Math.round(state.bestEpisodeReward),
        aliveCount,
        totalCount: CONFIG.NUM_ENVS,
        message: statusMessage,
        actorStats,
        criticStats,
        avgReward: avgReward,
        leaderProgress: leader ? leader.getDisplayProgress() : 0,
    });
    
    requestAnimationFrame(loop);
}

function ppoStep(car) {
    const stateVec = car.getStateVector(state.track);
    const wasInitialized = car.progressInitialized;
    const prevTotalProgress = car.totalProgress;
    
    // Get action from agent
    const { action, value, logProb } = state.agent.act(stateVec);
    
    // Apply action and update
    car.applyAction(action);
    car.update(state.track, state.cars);
    
    // Compute reward
    const effectivePrevProgress = wasInitialized ? prevTotalProgress : car.totalProgress;
    const reward = computeReward(car, effectivePrevProgress);
    car.episodeReward += reward;
    
    // Track discounted return for critic accuracy
    const discount = Math.pow(CONFIG.PPO.GAMMA, car.episodeLength);
    car.episodeDiscountedReturn += discount * reward;
    
    // Store in trajectory
    const done = car.dead || car.finished;
    car.trajectory.push({ state: stateVec, action, reward, value, logProb, done });
    
    // Debug logging
    debugLogger.logStep({
        state: stateVec,
        action,
        reward,
        value,
        progress: car.totalProgress,
        dead: car.dead,
    });
}

async function performPPOUpdate() {
    // Show training indicator and give browser a frame to render it
    showTrainingIndicator(true);
    await new Promise(r => setTimeout(r, 50));  // Let browser repaint
    
    const lastValues = state.cars.map(car => {
        if (car.dead || car.finished) return 0;
        return state.agent.getValue(car.getStateVector(state.track));
    });
    
    const stats = await state.agent.update(lastValues);
    state.updateCount = stats.updateCount;
    
    // Track average reward for chart
    const avgReward = average(state.recentRewards);
    pushWithLimit(state.rewardHistory, avgReward, state.maxHistoryLen);
    
    console.log(`📊 PPO Update #${stats.updateCount} - Avg reward: ${avgReward.toFixed(1)}`);
    
    debugLogger.logUpdate({
        updateCount: stats.updateCount,
        loss: stats.loss,
    });
    
    // Hide training indicator
    showTrainingIndicator(false);
}

function showTrainingIndicator(show) {
    const indicator = document.getElementById('training-indicator');
    const box = document.getElementById('learning-box');
    if (indicator) indicator.classList.toggle('hidden', !show);
    if (box) box.classList.toggle('training', show);
}

async function restartFromScratch() {
    console.log('🔄 Resetting training...');
    
    if (state.agent) {
        state.agent.dispose();
    }
    
    state.agent = new PPOAgent();
    await state.agent.pretrain();
    
    state.updateCount = 0;
    state.recentRewards = [];
    state.bestEpisodeReward = -Infinity;
    state.rewardHistory = [];
    state.criticPredictions = [];
    state.actualReturns = [];
    state.avgPrediction = 0;
    state.avgActual = 0;
    
    for (const car of state.cars) {
        car.reset(state.track.getStartLine());
    }
    
    console.log('✅ Training reset complete');
}

window.restartFromScratch = restartFromScratch;

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

window.addEventListener('load', setup);
