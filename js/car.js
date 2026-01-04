import { CONFIG } from './config.js';
import { rayCircleIntersection, getGridPosition, normalizeAngle } from './utils.js';

export class Car {
    constructor(id, x, y, angle) {
        this.id = id;
        this.startX = x;
        this.startY = y;
        this.startAngle = angle;
        
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.speed = 0;
        
        this.dead = false;
        this.finished = false;
        
        this.sensors = new Array(CONFIG.SENSOR_COUNT).fill(CONFIG.SENSOR_LENGTH);
        this.prevSensors = null;  // For TTC calculation (null until first real reading)
        this.sensorTypes = new Array(CONFIG.SENSOR_COUNT).fill('wall');
        
        this.color = `hsl(30, 80%, 50%)`;  // Default orange, will be updated by reward
        
        this.timer = 0;
        this.graceTimer = CONFIG.GRACE_PERIOD;
        
        this.rawProgress = 0;       // Raw spline t value (0-1)
        this.totalProgress = 0;      // Accumulated progress (can exceed 1.0 for multiple laps)
        this.progressInitialized = false;  // Track if first update has happened
        this.lapCount = 0;
        this.passedHalfway = false;
        this.progressVelocity = 0;
        this.justCompletedLap = false;  // Flag for reward computation
        this.progressInitialized = false;
        
        // Episode tracking
        this.episodeReward = 0;          // Undiscounted sum of rewards
        this.episodeDiscountedReturn = 0; // Discounted sum (what critic predicts)
        this.episodeLength = 0;
        this.criticPrediction = 0;       // V₀: what critic predicted at episode start
        
        // Episode trajectory for "complete episodes only" training
        this.trajectory = [];            // [{state, action, reward, value, logProb}, ...]
    }
    
    // Get normalized state vector for PPO
    getStateVector(track) {
        // Compute signed angle between car heading and track direction
        const trackDir = track.getTrackDirection(this.x, this.y);
        const angleToTrack = this.#computeSignedAngle(trackDir);
        
        return [
            // 8 LIDAR sensors normalized to [0, 1]
            ...this.sensors.map(s => s / CONFIG.SENSOR_LENGTH),
            // Speed normalized (approximate max ~12)
            this.speed / CONFIG.MAX_SPEED,
            // Signed angle to track direction, normalized to [-1, 1]
            // 0 = aligned, -1 = facing backward-left, +1 = facing backward-right
            angleToTrack / Math.PI,
        ];
    }
    
    // Compute signed angle between car heading and track tangent
    // Returns angle in [-PI, PI], where 0 = perfectly aligned with track
    #computeSignedAngle(trackTangent) {
        if (!trackTangent) return 0;
        const trackAngle = Math.atan2(trackTangent.y, trackTangent.x);
        return normalizeAngle(this.angle - trackAngle);
    }
    
    // Apply action from PPO agent
    applyAction(action) {
        const [steer] = action;
        
        // Hardcoded throttle: always forward
        const throttle = 1.0;
        
        // Apply physics
        this.speed += CONFIG.CAR_ACCEL * throttle;
        this.angle += steer * CONFIG.CAR_TURN_SPEED;
        this.speed *= CONFIG.CAR_FRICTION;
        this.speed = Math.max(0, this.speed);  // No reverse
        
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;
    }
    
    // Update sensors and check collisions (call after applyAction)
    update(track, otherCars) {
        if (this.dead || this.finished) return;
        
        this.timer++;
        this.episodeLength++;
        if (this.graceTimer > 0) this.graceTimer--;
        
        // Save previous sensors BEFORE update (for TTC calculation)
        // Only save if we have a valid reading (not first frame after reset)
        if (this.prevSensors !== null) {
            this.prevSensors = [...this.sensors];
        }
        
        // Update sensors
        this.#updateSensors(track, otherCars);
        
        // After first sensor update, initialize prevSensors for next frame
        if (this.prevSensors === null) {
            this.prevSensors = [...this.sensors];
        }
        
        // Check collisions
        this.#checkCollision(track, otherCars);
        
        // Update progress
        this.#updateProgress(track);
        
        // Check episode length limit
        if (this.timer >= CONFIG.MAX_EPISODE_LENGTH) {
            this.dead = true;
        }
    }
    
    #getSensorAngle(i) {
        // 7 forward-facing sensors spread across 180°, 1 rear sensor
        return (i < 7)
            ? this.angle - Math.PI / 2 + (i / 6) * Math.PI
            : this.angle + Math.PI;
    }
    
    #updateSensors(track, otherCars) {
        for (let i = 0; i < CONFIG.SENSOR_COUNT; i++) {
            const sensorAngle = this.#getSensorAngle(i);
            const dirX = Math.cos(sensorAngle);
            const dirY = Math.sin(sensorAngle);
            
            // Use SDF raycasting for road edge detection
            let minDist = track.raycastToEdge(this.x, this.y, dirX, dirY, CONFIG.SENSOR_LENGTH);
            this.sensorTypes[i] = 'wall';
            
            // Check other cars
            for (const car of otherCars) {
                if (car === this || car.dead) continue;
                
                const distToCarCenter = Math.hypot(this.x - car.x, this.y - car.y);
                if (distToCarCenter > CONFIG.SENSOR_LENGTH + CONFIG.CAR_SENSOR_RADIUS) continue;
                
                const intersectDist = rayCircleIntersection(
                    { x: this.x, y: this.y },
                    { x: dirX, y: dirY },
                    { x: car.x, y: car.y },
                    CONFIG.CAR_SENSOR_RADIUS
                );
                
                if (intersectDist !== null && intersectDist < minDist) {
                    minDist = intersectDist;
                    this.sensorTypes[i] = 'car';
                }
            }
            
            this.sensors[i] = minDist;
        }
    }
    
    #checkCollision(track, otherCars) {
        // Wall collision via SDF
        if (track.isOffRoad(this.x, this.y)) {
            this.die();
            return;
        }
        
        // Car-car collision
        if (this.graceTimer <= 0) {
            for (const car of otherCars) {
                if (car === this || car.dead || car.graceTimer > 0) continue;
                if (Math.hypot(this.x - car.x, this.y - car.y) < CONFIG.CAR_COLLISION_RADIUS) {
                    this.die();
                    car.die();
                    return;
                }
            }
        }
    }
    
    #updateProgress(track) {
        const rawProgress = track.getProgress(this.x, this.y);
        
        // Initialize on first update - normalize starting position
        // Cars start just behind start line, so raw might be ~0.98
        // We want to treat that as ~0 (or slightly negative)
        if (!this.progressInitialized) {
            this.rawProgress = rawProgress;
            // If starting near the end (e.g. 0.95-1.0), treat as slightly behind start
            if (rawProgress > 0.9) {
                this.totalProgress = rawProgress - 1.0;  // e.g., 0.98 -> -0.02
            } else {
                this.totalProgress = rawProgress;
            }
            this.progressInitialized = true;
            return;
        }
        
        // Compute delta progress (handles wraparound)
        let delta = rawProgress - this.rawProgress;
        if (delta > 0.5) delta -= 1;      // Wrapped backward (0.9 → 0.1 looks like +0.8, actually -0.2)
        if (delta < -0.5) delta += 1;     // Wrapped forward (0.1 → 0.9 looks like -0.8, actually +0.2)
        
        this.progressVelocity = delta;
        this.totalProgress += delta;
        
        // Track if car has passed halfway (within the current lap)
        const lapProgress = this.totalProgress % 1;
        if (lapProgress > 0.4 && lapProgress < 0.6) {
            this.passedHalfway = true;
        }
        
        // Detect lap completion (crossed from 0.9+ to next lap)
        const newLapCount = Math.floor(this.totalProgress);
        if (newLapCount > this.lapCount && this.passedHalfway) {
            this.lapCount = newLapCount;
            this.passedHalfway = false;
            this.justCompletedLap = true;
            if (this.lapCount >= 1) {
                this.finished = true;
            }
        }
        
        this.rawProgress = rawProgress;
    }
    
    // Get progress for display (0-100% within current lap attempt)
    getDisplayProgress() {
        // Show progress as percentage, clamped to 0-100
        const progress = Math.max(0, this.totalProgress);
        const lapProgress = progress % 1;
        return lapProgress;
    }
    
    // Set color based on normalized reward (0 = red, 1 = blue)
    // Uses a jet-like colormap: red -> yellow -> green -> cyan -> blue
    setRewardColor(normalizedReward) {
        // Clamp to [0, 1]
        const t = Math.max(0, Math.min(1, normalizedReward));
        
        // Jet colormap: hue goes from 0 (red) through 60 (yellow), 120 (green), 180 (cyan), to 240 (blue)
        // Map t=0 -> hue=0 (red), t=1 -> hue=240 (blue)
        const hue = t * 240;
        this.color = `hsl(${hue}, 85%, 50%)`;
    }
    
    // For backward compatibility
    get lastProgress() {
        return this.getDisplayProgress();
    }
    
    die() {
        this.dead = true;
        this.speed = 0;
    }
    
    // Reset car to specific position
    resetAt(x, y, angle) {
        this.x = x;
        this.y = y;
        this.angle = angle;
        
        this.speed = 0;
        this.dead = false;
        this.finished = false;
        this.timer = 0;
        this.graceTimer = CONFIG.GRACE_PERIOD;
        this.rawProgress = 0;
        this.totalProgress = 0;
        this.lapCount = 0;
        this.passedHalfway = false;
        this.progressVelocity = 0;
        this.justCompletedLap = false;
        this.progressInitialized = false;
        this.episodeReward = 0;
        this.episodeDiscountedReturn = 0;
        this.episodeLength = 0;
        this.criticPrediction = 0;
        this.trajectory = [];
        this.sensors.fill(CONFIG.SENSOR_LENGTH);
        this.prevSensors = null;
    }
    
    // Reset car to starting position (legacy - uses grid position)
    reset(startLine) {
        if (startLine) {
            const pos = getGridPosition(this.id, startLine, {
                lateralOffset: CONFIG.GRID_LATERAL_OFFSET,
                rowSpacing: CONFIG.GRID_ROW_SPACING,
                startOffset: CONFIG.GRID_START_OFFSET,
            });
            this.resetAt(pos.x, pos.y, pos.angle);
        } else {
            this.resetAt(this.startX, this.startY, this.startAngle);
        }
    }
    
    draw(ctx, isLeader) {
        // Draw sensors for leader
        if (isLeader && !this.dead) {
            this.sensors.forEach((dist, i) => {
                const angle = this.#getSensorAngle(i);
                ctx.beginPath();
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = this.sensorTypes[i] === 'car'
                    ? 'rgba(255, 255, 0, 0.9)'
                    : dist < 40 ? 'rgba(255, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.2)';
                ctx.moveTo(this.x, this.y);
                ctx.lineTo(this.x + Math.cos(angle) * dist, this.y + Math.sin(angle) * dist);
                ctx.stroke();
            });
        }
        
        // Draw car body
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);
        
        if (this.dead) {
            ctx.fillStyle = 'rgba(100, 100, 100, 0.4)';
        } else if (this.finished) {
            ctx.fillStyle = '#facc15';
        } else {
            // Use jet colormap color for all cars (including leader)
            ctx.fillStyle = this.color;
        }
        
        if (this.graceTimer > 0 && !this.dead) {
            ctx.globalAlpha = 0.6;
        }
        
        ctx.beginPath();
        ctx.roundRect(-15, -9, 30, 18, 4);
        ctx.fill();
        
        if (isLeader && !this.dead) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        ctx.restore();
    }
}
