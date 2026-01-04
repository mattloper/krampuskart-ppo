// Reward computation for PPO - Progress only
import { CONFIG } from '../config.js';

/**
 * Compute reward for a single step
 * Currently just rewards forward progress on the track
 */
export function computeReward(car, prevTotalProgress) {
    const deltaProgress = car.totalProgress - prevTotalProgress;
    const progressReward = deltaProgress * CONFIG.PPO.PROGRESS_WEIGHT;
    
    // Death penalty (if enabled)
    const deathPenalty = car.dead ? CONFIG.PPO.DEATH_PENALTY : 0;
    
    return progressReward + deathPenalty;
}
