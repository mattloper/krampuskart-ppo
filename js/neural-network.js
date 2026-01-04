// Activation functions
const leakyReLU = x => x > 0 ? x : 0.01 * x;

export class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes, weights = null) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        
        if (weights) {
            this.weights1 = weights.w1;
            this.weights2 = weights.w2;
        } else {
            this.weights1 = this.#randomMatrix(hiddenNodes, inputNodes);
            this.weights2 = this.#randomMatrix(outputNodes, hiddenNodes);
        }
    }
    
    #randomMatrix(rows, cols) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }
    
    mutate(rate) {
        const mutateVal = (val) => 
            Math.random() < rate ? val + (Math.random() * 0.4 - 0.2) : val;
        
        this.weights1 = this.weights1.map(row => row.map(mutateVal));
        this.weights2 = this.weights2.map(row => row.map(mutateVal));
    }
    
    predict(inputs) {
        // Normalize inputs:
        // - First 8: sensor distances (0-200) → divide by 200
        // - Last 1: progress velocity (already scaled ~-10 to +10) → divide by 200 for similar range
        const normalized = inputs.map(v => v / 200);
        
        // Hidden layer (Leaky ReLU - doesn't saturate like tanh)
        const hidden = this.weights1.map(row => {
            let sum = 0;
            for (let i = 0; i < normalized.length; i++) {
                sum += normalized[i] * row[i];
            }
            return leakyReLU(sum);
        });
        
        // Output layer (tanh for bounded [-1, 1] steering/throttle)
        const output = this.weights2.map(row => {
            let sum = 0;
            for (let i = 0; i < hidden.length; i++) {
                sum += hidden[i] * row[i];
            }
            return Math.tanh(sum);
        });
        
        return output;
    }
    
    copy() {
        return new NeuralNetwork(
            this.inputNodes,
            this.hiddenNodes,
            this.outputNodes,
            {
                w1: this.weights1.map(row => [...row]),
                w2: this.weights2.map(row => [...row])
            }
        );
    }
    
    getWeightStats() {
        const allWeights = [...this.weights1.flat(), ...this.weights2.flat()];
        const mean = allWeights.reduce((a, b) => a + b, 0) / allWeights.length;
        const std = Math.sqrt(
            allWeights.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / allWeights.length
        );
        return { mean, std };
    }
    
    draw(ctx, width, height) {
        ctx.clearRect(0, 0, width, height);
        
        const layers = [this.inputNodes, this.hiddenNodes, this.outputNodes];
        const layerX = [20, width / 2, width - 20];
        const nodeRadius = 4;
        
        // Draw weights as lines (use raw weight value, not z-score)
        const drawWeights = (wMatrix, x1, x2, layer1Count, layer2Count) => {
            for (let i = 0; i < layer1Count; i++) {
                for (let j = 0; j < layer2Count; j++) {
                    const weight = wMatrix[j][i];
                    // Use raw weight: positive = green, negative = red
                    // Clamp to [-2, 2] range for alpha calculation
                    const clamped = Math.max(-2, Math.min(2, weight));
                    const alpha = 0.15 + Math.abs(clamped) * 0.35;  // Always visible
                    
                    ctx.beginPath();
                    ctx.strokeStyle = weight > 0 
                        ? `rgba(0, 255, 100, ${alpha})` 
                        : `rgba(255, 50, 50, ${alpha})`;
                    ctx.lineWidth = 0.5 + Math.abs(clamped) * 0.5;
                    
                    const y1 = (height / (layer1Count + 1)) * (i + 1);
                    const y2 = (height / (layer2Count + 1)) * (j + 1);
                    
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            }
        };
        
        drawWeights(this.weights1, layerX[0], layerX[1], this.inputNodes, this.hiddenNodes);
        drawWeights(this.weights2, layerX[1], layerX[2], this.hiddenNodes, this.outputNodes);
        
        // Draw nodes
        layers.forEach((count, lIdx) => {
            for (let i = 0; i < count; i++) {
                const y = (height / (count + 1)) * (i + 1);
                ctx.beginPath();
                ctx.fillStyle = '#fff';
                ctx.arc(layerX[lIdx], y, nodeRadius, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }
    
    static weightedAverage(brains, weightsArray) {
        if (!brains || brains.length === 0) return null;
        
        const input = brains[0].inputNodes;
        const hidden = brains[0].hiddenNodes;
        const output = brains[0].outputNodes;
        
        const combine = (matrixKey, rowIdx, colIdx) => {
            let sum = 0;
            for (let i = 0; i < brains.length; i++) {
                sum += brains[i][matrixKey][rowIdx][colIdx] * weightsArray[i];
            }
            return sum;
        };
        
        const avgW1 = Array.from({ length: hidden }, (_, rowIdx) =>
            Array.from({ length: input }, (_, colIdx) => combine('weights1', rowIdx, colIdx))
        );
        
        const avgW2 = Array.from({ length: output }, (_, rowIdx) =>
            Array.from({ length: hidden }, (_, colIdx) => combine('weights2', rowIdx, colIdx))
        );
        
        return new NeuralNetwork(input, hidden, output, { w1: avgW1, w2: avgW2 });
    }
}

