// Actor-Critic Networks using TensorFlow.js
// SEPARATE networks for actor (policy) and critic (value) - as recommended by PPO paper for continuous control

let modelCounter = 0;

export class ActorCritic {
    constructor(inputDim, actionDim, hiddenUnits = [64, 64]) {
        this.inputDim = inputDim;
        this.actionDim = actionDim;
        this.hiddenUnits = hiddenUnits;
        this.modelId = modelCounter++;
        
        // Build SEPARATE actor and critic networks (no shared backbone)
        this._buildActorNetwork();
        this._buildCriticNetwork();
        
        // Learnable log standard deviation for actor
        this.logStd = tf.variable(tf.fill([actionDim], -1.0), true, `logStd_${this.modelId}`);
    }
    
    // GELU activation layer
    _gelu(x, name) {
        class GELULayer extends tf.layers.Layer {
            constructor(config) {
                super(config);
            }
            call(inputs) {
                return tf.tidy(() => {
                    const x = Array.isArray(inputs) ? inputs[0] : inputs;
                    const cdf = tf.mul(0.5, tf.add(1, tf.tanh(
                        tf.mul(Math.sqrt(2 / Math.PI), tf.add(x, tf.mul(0.044715, tf.pow(x, 3))))
                    )));
                    return tf.mul(x, cdf);
                });
            }
            static get className() { return 'GELULayer'; }
        }
        return new GELULayer({ name }).apply(x);
    }
    
    _buildActorNetwork() {
        const input = tf.input({ shape: [this.inputDim] });
        
        // Actor's own hidden layers
        let x = input;
        for (let i = 0; i < this.hiddenUnits.length; i++) {
            x = tf.layers.dense({ 
                units: this.hiddenUnits[i], 
                activation: 'linear',
                kernelInitializer: 'heNormal',
                name: `actor_dense${i + 1}_${this.modelId}`
            }).apply(x);
            x = this._gelu(x, `actor_gelu${i + 1}_${this.modelId}`);
        }
        
        // Actor output - mean of Gaussian policy
        const actorMean = tf.layers.dense({ 
            units: this.actionDim, 
            activation: 'tanh',
            kernelInitializer: tf.initializers.randomUniform({ minval: -0.03, maxval: 0.03 }),
            name: `actor_out_${this.modelId}`
        }).apply(x);
        
        this.actorModel = tf.model({ 
            inputs: input, 
            outputs: actorMean,
            name: `actor_${this.modelId}`
        });
    }
    
    _buildCriticNetwork() {
        const input = tf.input({ shape: [this.inputDim] });
        
        // Critic's own hidden layers
        let x = input;
        for (let i = 0; i < this.hiddenUnits.length; i++) {
            x = tf.layers.dense({ 
                units: this.hiddenUnits[i], 
                activation: 'linear',
                kernelInitializer: 'heNormal',
                name: `critic_dense${i + 1}_${this.modelId}`
            }).apply(x);
            x = this._gelu(x, `critic_gelu${i + 1}_${this.modelId}`);
        }
        
        // Critic output - state value
        const criticValue = tf.layers.dense({ 
            units: 1, 
            activation: 'linear',
            kernelInitializer: 'glorotUniform',
            name: `critic_out_${this.modelId}`
        }).apply(x);
        
        this.criticModel = tf.model({ 
            inputs: input, 
            outputs: criticValue,
            name: `critic_${this.modelId}`
        });
    }
    
    // Forward pass - returns action mean and value from separate networks
    forward(states) {
        const mean = this.actorModel.predict(states);
        const value = this.criticModel.predict(states);
        return { 
            mean, 
            value: tf.squeeze(value, -1),
            logStd: this.logStd
        };
    }
    
    // Sample action from Gaussian policy (for collecting experience)
    act(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const meanTensor = this.actorModel.predict(stateTensor);
            const valueTensor = this.criticModel.predict(stateTensor);
            
            const mean = meanTensor.dataSync();
            const value = valueTensor.dataSync()[0];
            const logStdArr = this.logStd.dataSync();
            
            // Sample from Gaussian
            const action = [];
            let logProb = 0;
            
            for (let i = 0; i < this.actionDim; i++) {
                const std = Math.exp(logStdArr[i]);
                const noise = this._randn();
                const a = Math.max(-1, Math.min(1, mean[i] + std * noise));
                action.push(a);
                
                // Log probability
                const diff = a - mean[i];
                logProb += -0.5 * (diff * diff / (std * std) + 2 * logStdArr[i] + Math.log(2 * Math.PI));
            }
            
            return { action, value, logProb, mean: Array.from(mean) };
        });
    }
    
    // Get value only (for bootstrapping)
    getValue(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const valueTensor = this.criticModel.predict(stateTensor);
            return valueTensor.dataSync()[0];
        });
    }
    
    // Standard normal random
    _randn() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
    
    // Compute log probability for a batch (for PPO update)
    computeLogProb(means, actions) {
        const std = tf.exp(this.logStd);
        const variance = tf.square(std);
        
        const diff = tf.sub(actions, means);
        const logProb = tf.mul(-0.5, tf.add(
            tf.div(tf.square(diff), variance),
            tf.add(tf.mul(2, this.logStd), Math.log(2 * Math.PI))
        ));
        
        return tf.sum(logProb, -1);
    }
    
    // Get entropy of policy (as a scalar for logging)
    getEntropy() {
        const logStdArr = this.logStd.dataSync();
        let entropy = 0;
        for (let i = 0; i < this.actionDim; i++) {
            entropy += 0.5 + 0.5 * Math.log(2 * Math.PI) + logStdArr[i];
        }
        return entropy;
    }
    
    // Get entropy as a tensor (for including in loss)
    getEntropyTensor() {
        const perDim = tf.add(0.5 + 0.5 * Math.log(2 * Math.PI), this.logStd);
        return tf.sum(perDim);
    }
    
    // Get logStd values for display
    getLogStdValues() {
        return Array.from(this.logStd.dataSync());
    }
    
    // Get trainable weights for actor (for policy optimization)
    getActorTrainableWeights() {
        return this.actorModel.trainableWeights;
    }
    
    // Get trainable weights for critic (for value optimization)
    getCriticTrainableWeights() {
        return this.criticModel.trainableWeights;
    }
    
    // Pretrain actor with simple heuristic: counter-steer angle error
    async pretrain(numSamples = 500, epochs = 20) {
        console.log('🎓 Pretraining actor with behavioral cloning...');
        
        const states = [];
        const targetActions = [];
        
        for (let i = 0; i < numSamples; i++) {
            const sensors = Array.from({ length: 8 }, () => Math.random());
            const speed = Math.random();
            const angleToTrack = (Math.random() - 0.5) * 2;
            
            const state = [...sensors, speed, angleToTrack];
            const steering = -angleToTrack * 0.8;
            
            states.push(state);
            targetActions.push([Math.max(-1, Math.min(1, steering))]);
        }
        
        const statesTensor = tf.tensor2d(states);
        const actionsTensor = tf.tensor2d(targetActions);
        
        const optimizer = tf.train.adam(0.001);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            const loss = optimizer.minimize(() => {
                const meanPred = this.actorModel.predict(statesTensor);
                return tf.losses.meanSquaredError(actionsTensor, meanPred);
            }, true);
            
            if (epoch % 5 === 0) {
                const lossVal = loss.dataSync()[0];
                console.log(`  Pretrain epoch ${epoch + 1}/${epochs}, loss: ${lossVal.toFixed(4)}`);
            }
            loss.dispose();
        }
        
        statesTensor.dispose();
        actionsTensor.dispose();
        
        console.log('✅ Actor pretraining complete!');
    }
    
    dispose() {
        if (this.logStd) {
            this.logStd.dispose();
        }
        if (this.actorModel) {
            this.actorModel.dispose();
        }
        if (this.criticModel) {
            this.criticModel.dispose();
        }
    }
}
