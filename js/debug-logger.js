// Debug logger - stores training data for analysis
import { CONFIG } from './config.js';
import { average } from './utils.js';

class DebugLogger {
    constructor() {
        this.logs = [];
        this.stepCount = 0;
        this.maxLogs = 10000; // Keep last N entries
    }
    
    logStep(data) {
        if (!CONFIG.DEBUG_LOG) return;
        
        this.stepCount++;
        const entry = {
            step: this.stepCount,
            time: Date.now(),
            ...data
        };
        
        this.logs.push(entry);
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }
        
        // Console output every 100 steps
        if (this.stepCount % 100 === 0) {
            console.log(`[Step ${this.stepCount}]`, this.formatForConsole(data));
        }
    }
    
    logUpdate(data) {
        if (!CONFIG.DEBUG_LOG) return;
        
        console.log('═'.repeat(60));
        console.log('PPO UPDATE');
        console.log('═'.repeat(60));
        
        if (data.loss) {
            console.log(`  Policy Loss: ${data.loss.policy?.toFixed(6) || 'N/A'}`);
            console.log(`  Value Loss:  ${data.loss.value?.toFixed(6) || 'N/A'}`);
            console.log(`  Total Loss:  ${data.loss.total?.toFixed(6) || 'N/A'}`);
        }
        
        if (data.gradients !== undefined) {
            console.log(`  Gradients:   ${data.gradients ? '✓ flowing' : '✗ NOT flowing'}`);
            console.log(`  Weight Δ:    ${data.weightDiff?.toExponential(3) || 'N/A'}`);
        }
        
        if (data.meanReward !== undefined) {
            console.log(`  Mean Reward: ${data.meanReward.toFixed(4)}`);
        }
        
        console.log('═'.repeat(60));
        
        this.logs.push({
            step: this.stepCount,
            type: 'update',
            time: Date.now(),
            ...data
        });
    }
    
    formatForConsole(data) {
        const parts = [];
        
        if (data.state) {
            const s = data.state;
            parts.push(`IN:[${s.slice(0,4).map(v => v.toFixed(2)).join(',')}...] spd:${s[8]?.toFixed(2)} ang:${s[9]?.toFixed(2)}`);
        }
        
        if (data.action) {
            parts.push(`OUT:[steer:${data.action[0]?.toFixed(3)} thr:${data.action[1]?.toFixed(3)}]`);
        }
        
        if (data.reward !== undefined) {
            parts.push(`R:${data.reward.toFixed(4)}`);
        }
        
        if (data.value !== undefined) {
            parts.push(`V:${data.value.toFixed(4)}`);
        }
        
        if (data.progress !== undefined) {
            parts.push(`prog:${(data.progress * 100).toFixed(1)}%`);
        }
        
        return parts.join(' | ');
    }
    
    // Get recent summary
    getSummary(n = 100) {
        const recent = this.logs.slice(-n);
        const steps = recent.filter(l => l.type !== 'update');
        const updates = recent.filter(l => l.type === 'update');
        
        if (steps.length === 0) return null;
        
        const avgReward = average(steps.map(l => l.reward || 0));
        const avgValue = average(steps.map(l => l.value || 0));
        
        return {
            totalSteps: this.stepCount,
            recentSteps: steps.length,
            avgReward: avgReward.toFixed(4),
            avgValue: avgValue.toFixed(4),
            updates: updates.length
        };
    }
    
    clear() {
        this.logs = [];
        this.stepCount = 0;
        console.log('Debug logs cleared');
    }
}

export const debugLogger = new DebugLogger();

// Expose to window for console access
window.debugLogger = debugLogger;
window.logSummary = () => console.table(debugLogger.getSummary());

