// Module load ID - random on each fresh load, same if cached
export const MODULE_ID = Math.random().toString(36).substring(2, 8);
export const LOAD_TIME = new Date().toLocaleTimeString();

export const CONFIG = {
    // Parallel environments (cars)
    NUM_ENVS: 24,
    
    // Debug logging
    DEBUG_LOG: true,
    
    // Sensors
    SENSOR_COUNT: 8,
    SENSOR_LENGTH: 600,      // Tripled from 200 for longer range
    
    // Track
    ROAD_HALF_WIDTH: 135,
    TRACK_SAMPLES: 200,
    
    // Physics
    CAR_ACCEL: 0.44,         // Doubled throttle speed
    CAR_FRICTION: 0.96,
    CAR_TURN_SPEED: 0.24,      // Doubled for even sharper turns
    CAR_COLLISION_RADIUS: 22,
    CAR_SENSOR_RADIUS: 15,
    MAX_SPEED: 12,  // Approximate max speed for normalization
    
    // Timers
    GRACE_PERIOD: 60,
    MAX_EPISODE_LENGTH: Infinity,  // No timeout - only die from collisions
    
    // Camera
    CAMERA_SMOOTHING: 0.1,
    
    // Grid spawning
    GRID_LATERAL_OFFSET: 35,
    GRID_ROW_SPACING: 50,
    GRID_START_OFFSET: 30,
    
    // Random spawn config
    SPAWN_LATERAL_SPREAD: 60,
    SPAWN_LONGITUDINAL_SPREAD: 200,
    
    // PPO Hyperparameters
    PPO: {
        // Network architecture
        INPUT_DIM: 10,           // 8 sensors + speed + angle to track
        ACTION_DIM: 1,           // steering only (throttle hardcoded)
        HIDDEN_UNITS: [4],       // Tiny network - just 4 hidden units!
        
        // PPO algorithm
        GAMMA: 0.995,            // Discount factor (longer horizon)
        GAE_LAMBDA: 0.95,        // GAE lambda for advantage estimation
        CLIP_EPSILON: 0.1,       // Tighter clip for stability
        ENTROPY_COEF: 0.01,      // Entropy bonus coefficient
        VALUE_COEF: 10.0,        // Very high to force critic learning
        
        // Training
        LEARNING_RATE: 3e-4,  // Fast learning rate
        BATCH_SIZE: 64,
        EPOCHS_PER_UPDATE: 10,   // More epochs since we only train on complete episodes
        MIN_EPISODES_FOR_UPDATE: 20, // Train after this many complete episodes
        SUBSAMPLE_RATIO: 1,      // Use all samples
        
        // Reward
        PROGRESS_WEIGHT: 500,         // deltaProgress * weight
        DEATH_PENALTY: 0,             // Penalty on crash (0 = disabled)
    }
};

