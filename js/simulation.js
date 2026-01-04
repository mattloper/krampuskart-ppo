// Simulation logic for the racing environment
// Pure functions that don't depend on global state

import { Car } from './car.js';
import { getGridPosition } from './utils.js';
import { CONFIG } from './config.js';

/**
 * Generate a random valid spawn position near the start line
 * @param {Track} track - The track object
 * @param {Object} startLine - { point, tangent, normal }
 * @param {Object} spawnConfig - { lateralSpread, longitudinalSpread }
 * @returns {{ x, y, angle }} Valid spawn position
 */
function getRandomSpawnPosition(track, startLine, spawnConfig = {}) {
    const maxAttempts = 50;
    const lateralSpread = spawnConfig.lateralSpread ?? CONFIG.SPAWN_LATERAL_SPREAD;
    const longitudinalSpread = spawnConfig.longitudinalSpread ?? CONFIG.SPAWN_LONGITUDINAL_SPREAD;
    
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        // Random offset along track direction (behind start line)
        const longitudinal = -Math.random() * longitudinalSpread;
        // Random lateral offset (left/right of center)
        const lateral = (Math.random() - 0.5) * 2 * lateralSpread;
        
        const x = startLine.point.x 
            + startLine.tangent.x * longitudinal 
            + startLine.normal.x * lateral;
        const y = startLine.point.y 
            + startLine.tangent.y * longitudinal 
            + startLine.normal.y * lateral;
        const angle = Math.atan2(startLine.tangent.y, startLine.tangent.x);
        
        // Check if position is on track
        if (!track.isOffRoad(x, y)) {
            return { x, y, angle };
        }
    }
    
    // Fallback: return center of start line
    console.warn('Could not find valid spawn position after', maxAttempts, 'attempts');
    return {
        x: startLine.point.x - startLine.tangent.x * 50,
        y: startLine.point.y - startLine.tangent.y * 50,
        angle: Math.atan2(startLine.tangent.y, startLine.tangent.x)
    };
}

/**
 * Create initial car instances on the track
 * @param {Track} track - The track object
 * @param {number} numCars - Number of cars to spawn
 * @param {Object} spawnConfig - { lateralSpread, longitudinalSpread }
 * @returns {Car[]} Array of spawned cars
 */
export function spawnCars(track, numCars, spawnConfig = {}) {
    const cars = [];
    const startLine = track.getStartLine();
    
    for (let i = 0; i < numCars; i++) {
        const pos = getRandomSpawnPosition(track, startLine, spawnConfig);
        const car = new Car(i, pos.x, pos.y, pos.angle);
        cars.push(car);
    }
    
    return cars;
}

/**
 * Find the leading car (highest progress) and count alive cars
 * @param {Car[]} cars - Array of cars
 * @returns {{ leader: Car|null, aliveCount: number }}
 */
export function findLeader(cars) {
    let leader = null;
    let maxProgress = -Infinity;
    let aliveCount = 0;
    
    for (const car of cars) {
        if (!car.dead) {
            aliveCount++;
            const progress = car.lapCount + car.lastProgress;
            if (progress > maxProgress) {
                maxProgress = progress;
                leader = car;
            }
        }
    }
    
    // Fallback to first car if all dead
    if (!leader && cars.length > 0) {
        leader = cars[0];
    }
    
    return { leader, aliveCount };
}

/**
 * Process finished/dead cars - record stats and reset them
 * @param {Car[]} cars - Array of cars
 * @param {Track} track - The track object (for validation)
 * @param {Function} onEpisodeEnd - Callback(car) when an episode ends
 */
export function resetFinishedCars(cars, track, onEpisodeEnd) {
    const startLine = track.getStartLine();
    
    for (const car of cars) {
        if (car.dead || car.finished) {
            // Call callback before reset so it can read episode data
            if (onEpisodeEnd) {
                onEpisodeEnd(car);
            }
            
            // Get a valid random spawn position (uses CONFIG defaults)
            const pos = getRandomSpawnPosition(track, startLine);
            
            // Reset car at new random position
            car.resetAt(pos.x, pos.y, pos.angle);
        }
    }
}

/**
 * Smoothly update camera position toward target
 * @param {Object} camera - { x, y } camera position (mutated)
 * @param {Object} target - { x, y } target position
 * @param {number} smoothing - Smoothing factor (0-1)
 */
export function updateCamera(camera, target, smoothing) {
    camera.x += (target.x - camera.x) * smoothing;
    camera.y += (target.y - camera.y) * smoothing;
}

/**
 * Calculate steps per second with smoothing
 * @param {Object} timing - { lastUpdateTime, totalSteps } (mutated)
 * @param {number} now - Current time from performance.now()
 * @param {number} updateInterval - Minimum seconds between updates (e.g. 0.5)
 * @returns {number|null} - Steps per second, or null if not enough time elapsed
 */
export function calculateStepsPerSecond(timing, now, updateInterval = 0.5) {
    const dt = (now - timing.lastUpdateTime) / 1000;
    if (dt > updateInterval) {
        const stepsPerSecond = Math.round(timing.totalSteps / dt);
        timing.lastUpdateTime = now;
        timing.totalSteps = 0;
        return stepsPerSecond;
    }
    return null;
}
