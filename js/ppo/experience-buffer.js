// Experience Buffer for PPO
// Stores rollout data and computes GAE (Generalized Advantage Estimation)

import { average } from '../utils.js';

export class ExperienceBuffer {
    constructor() {
        this.clear();
    }
    
    clear() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.values = [];
        this.logProbs = [];
        this.dones = [];
        this.returns = null;
        this.advantages = null;
    }
    
    get length() {
        return this.states.length;
    }
    
    add(state, action, reward, value, logProb, done) {
        this.states.push(state);
        this.actions.push(action);
        this.rewards.push(reward);
        this.values.push(value);
        this.logProbs.push(logProb);
        this.dones.push(done ? 1 : 0);
    }
    
    // Compute returns and advantages
    // Uses Monte Carlo returns for critic (actual discounted rewards, no bootstrap pollution)
    // Uses GAE for advantages (actor still benefits from variance reduction)
    computeReturnsAndAdvantages(lastValue, gamma, lambda) {
        const n = this.rewards.length;
        this.returns = new Array(n);       // Monte Carlo returns for critic
        this.advantages = new Array(n);
        
        // STEP 1: Compute MONTE CARLO returns (no bootstrap pollution)
        // This gives the critic TRUE targets based on actual observed rewards
        let mcReturn = lastValue;  // Bootstrap only at the very end of rollout
        for (let t = n - 1; t >= 0; t--) {
            if (this.dones[t]) {
                // Episode ended here - return is just this step's reward
                mcReturn = this.rewards[t];
            } else {
                // Accumulate discounted reward
                mcReturn = this.rewards[t] + gamma * mcReturn;
            }
            this.returns[t] = mcReturn;
        }
        
        // STEP 2: Compute GAE advantages for the actor
        // Actor still benefits from variance reduction of GAE
        let gae = 0;
        for (let t = n - 1; t >= 0; t--) {
            const nextValue = (t === n - 1) ? lastValue : this.values[t + 1];
            
            // TD error: δ = r + γ * V(s') * (1 - done) - V(s)
            const delta = this.rewards[t] + gamma * nextValue * (1 - this.dones[t]) - this.values[t];
            
            // GAE: A = δ + γλ * (1 - done) * A'
            gae = delta + gamma * lambda * (1 - this.dones[t]) * gae;
            
            this.advantages[t] = gae;
        }
        
        // Compute statistics
        const avgReturn = average(this.returns);
        const avgValue = average(this.values);
        const avgReward = average(this.rewards);
        
        // Store for external access (so UI can compare V₀ to actual discounted return)
        this.lastAvgMCReturn = avgReturn;
        
        console.log(`📊 MC Returns: avgReturn=${avgReturn.toFixed(2)}, avgValue=${avgValue.toFixed(2)}, avgReward=${avgReward.toFixed(3)}, n=${n}`);
        
        // Normalize advantages
        this._normalizeAdvantages();
    }
    
    _normalizeAdvantages() {
        if (this.advantages.length === 0) return;
        
        const mean = average(this.advantages);
        const variance = this.advantages.reduce((a, b) => a + (b - mean) ** 2, 0) / this.advantages.length;
        const std = Math.sqrt(variance) + 1e-8;
        
        this.advantages = this.advantages.map(a => (a - mean) / std);
    }
    
    // Get shuffled mini-batches for training
    // subsampleRatio: only use every Nth sample to reduce redundancy
    getBatches(batchSize, subsampleRatio = 1) {
        const n = this.states.length;
        
        // Subsample: only use every Nth index to reduce correlated samples
        let indices = [];
        if (subsampleRatio > 1) {
            // Take every Nth sample
            for (let i = 0; i < n; i += subsampleRatio) {
                indices.push(i);
            }
        } else {
            indices = Array.from({ length: n }, (_, i) => i);
        }
        
        // Shuffle the (subsampled) indices
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        // Generate batches
        const batches = [];
        for (let start = 0; start < indices.length; start += batchSize) {
            const batchIndices = indices.slice(start, start + batchSize);
            if (batchIndices.length < batchSize / 2) continue; // Skip tiny batches
            
            batches.push({
                states: batchIndices.map(i => this.states[i]),
                actions: batchIndices.map(i => this.actions[i]),
                returns: batchIndices.map(i => this.returns[i]),
                advantages: batchIndices.map(i => this.advantages[i]),
                oldLogProbs: batchIndices.map(i => this.logProbs[i]),
            });
        }
        
        return batches;
    }
    
    // Statistics for logging
    getStats() {
        const n = this.rewards.length;
        if (n === 0) return { meanReward: 0, totalReward: 0, episodes: 0 };
        
        const totalReward = this.rewards.reduce((a, b) => a + b, 0);
        const episodes = this.dones.reduce((a, b) => a + b, 0);
        
        return {
            meanReward: totalReward / n,
            totalReward,
            episodes,
            steps: n
        };
    }
}


