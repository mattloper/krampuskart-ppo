// Neural Network Visualization
// Displays SEPARATE actor and critic networks with jet-colored weights

/**
 * Jet colormap: maps value [0,1] to RGB
 * 0 = red, 0.5 = green, 1 = blue
 */
function jetColor(t) {
    t = Math.max(0, Math.min(1, t));
    
    let r, g, b;
    
    if (t < 0.25) {
        r = 255;
        g = Math.round(t * 4 * 255);
        b = 0;
    } else if (t < 0.5) {
        r = Math.round((0.5 - t) * 4 * 255);
        g = 255;
        b = 0;
    } else if (t < 0.75) {
        r = 0;
        g = 255;
        b = Math.round((t - 0.5) * 4 * 255);
    } else {
        r = 0;
        g = Math.round((1 - t) * 4 * 255);
        b = 255;
    }
    
    return `rgb(${r},${g},${b})`;
}

/**
 * Extract weights from a single network
 */
function extractWeightsFromModel(model) {
    const layers = [];
    const weights = model.getWeights();
    
    let weightIdx = 0;
    for (const layer of model.layers) {
        if (layer.getWeights().length > 0) {
            const kernel = weights[weightIdx];
            const bias = weights[weightIdx + 1];
            
            layers.push({
                name: layer.name,
                weights: kernel.arraySync(),
                biases: bias ? bias.arraySync() : null,
                inputSize: kernel.shape[0],
                outputSize: kernel.shape[1]
            });
            
            weightIdx += 2;
        }
    }
    
    return layers;
}

/**
 * Extract network info from SEPARATE actor and critic models
 */
export function extractNetworkInfo(actorCritic) {
    return {
        actor: extractWeightsFromModel(actorCritic.actorModel),
        critic: extractWeightsFromModel(actorCritic.criticModel)
    };
}

/**
 * Draw a single network (actor or critic)
 */
function drawSingleNetwork(ctx, x, y, width, height, layers, title, titleColor, outputLabel) {
    if (!layers || layers.length === 0) return;
    
    const padding = { top: 25, bottom: 20, left: 15, right: 15 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;
    
    // Title
    ctx.fillStyle = titleColor;
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(title, x + width / 2, y + 12);
    
    // Find global weight range
    let minWeight = Infinity, maxWeight = -Infinity;
    for (const layer of layers) {
        for (const row of layer.weights) {
            for (const w of row) {
                minWeight = Math.min(minWeight, w);
                maxWeight = Math.max(maxWeight, w);
            }
        }
    }
    const weightRange = Math.max(Math.abs(minWeight), Math.abs(maxWeight)) || 1;
    
    // Calculate column positions
    const numColumns = layers.length + 1;  // Input + each layer
    const colSpacing = chartW / (numColumns);
    
    const nodeRadius = 3;
    const maxNodesDisplay = 20;
    
    // Create node positions for a layer
    const createNodePositions = (colX, size) => {
        const displaySize = Math.min(size, maxNodesDisplay);
        const nodeSpacing = chartH / (displaySize + 1);
        const positions = [];
        
        for (let n = 0; n < displaySize; n++) {
            positions.push({ 
                x: colX, 
                y: y + padding.top + (n + 1) * nodeSpacing, 
                index: n, 
                total: size 
            });
        }
        
        return positions;
    };
    
    // Input layer
    const inputSize = layers[0].inputSize;
    const inputX = x + padding.left + colSpacing * 0.5;
    const inputNodes = createNodePositions(inputX, inputSize);
    
    // Hidden layers + output
    const allLayerNodes = [];
    for (let i = 0; i < layers.length; i++) {
        const colX = x + padding.left + colSpacing * (i + 1.5);
        const nodes = createNodePositions(colX, layers[i].outputSize);
        allLayerNodes.push(nodes);
    }
    
    // Draw connections
    ctx.globalAlpha = 0.5;
    
    const drawConnections = (fromNodes, toNodes, weights) => {
        for (const fromNode of fromNodes) {
            for (const toNode of toNodes) {
                const wRow = Math.min(fromNode.index, weights.length - 1);
                const wCol = Math.min(toNode.index, weights[0].length - 1);
                const weight = weights[wRow][wCol];
                
                const normalized = (weight / weightRange + 1) / 2;
                
                ctx.strokeStyle = jetColor(normalized);
                ctx.lineWidth = Math.abs(weight / weightRange) * 1.5 + 0.3;
                
                ctx.beginPath();
                ctx.moveTo(fromNode.x, fromNode.y);
                ctx.lineTo(toNode.x, toNode.y);
                ctx.stroke();
            }
        }
    };
    
    // Input -> first layer
    drawConnections(inputNodes, allLayerNodes[0], layers[0].weights);
    
    // Layer -> layer
    for (let i = 1; i < layers.length; i++) {
        drawConnections(allLayerNodes[i - 1], allLayerNodes[i], layers[i].weights);
    }
    
    ctx.globalAlpha = 1;
    
    // Draw nodes
    const drawNodes = (nodes) => {
        for (const node of nodes) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
    };
    
    drawNodes(inputNodes);
    allLayerNodes.forEach(nodes => drawNodes(nodes));
    
    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    
    ctx.fillText(`In(${inputSize})`, inputX, y + height - 5);
    
    for (let i = 0; i < allLayerNodes.length; i++) {
        const colX = x + padding.left + colSpacing * (i + 1.5);
        const isOutput = i === allLayerNodes.length - 1;
        const label = isOutput ? outputLabel : `H${i + 1}(${layers[i].outputSize})`;
        ctx.fillStyle = isOutput ? titleColor : '#888';
        ctx.fillText(label, colX, y + height - 5);
    }
}

/**
 * Draw both actor and critic networks side by side
 */
export function drawNNVisualization(ctx, width, height, networkInfo) {
    ctx.clearRect(0, 0, width, height);
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(0, 0, width, height);
    
    if (!networkInfo || !networkInfo.actor || !networkInfo.critic) {
        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.fillText('No network data', 10, height / 2);
        return;
    }
    
    const halfWidth = width / 2;
    
    // Draw Actor network (left side)
    drawSingleNetwork(ctx, 0, 0, halfWidth - 2, height - 15, 
        networkInfo.actor, 'ACTOR (Policy)', '#4ade80', 'Steer');
    
    // Draw Critic network (right side)
    drawSingleNetwork(ctx, halfWidth + 2, 0, halfWidth - 2, height - 15, 
        networkInfo.critic, 'CRITIC (Value)', '#60a5fa', 'Value');
    
    // Divider line
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(halfWidth, 20);
    ctx.lineTo(halfWidth, height - 20);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Color legend at bottom
    const barW = 60, barH = 5;
    const barX = (width - barW) / 2;
    const barY = height - 10;
    
    for (let i = 0; i < barW; i++) {
        ctx.fillStyle = jetColor(i / barW);
        ctx.fillRect(barX + i, barY, 1, barH);
    }
    
    ctx.fillStyle = '#888';
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('−', barX - 8, barY + 4);
    ctx.fillText('+', barX + barW + 8, barY + 4);
}
