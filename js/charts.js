// Chart visualization for PPO training

/**
 * Draw average reward over time
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} width
 * @param {number} height
 * @param {number[]} data - Array of average rewards per PPO update
 */
export function drawRewardChart(ctx, width, height, data) {
    ctx.clearRect(0, 0, width, height);
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, width, height);
    
    if (!data || data.length < 2) {
        ctx.fillStyle = '#666';
        ctx.font = '11px monospace';
        ctx.fillText('Collecting reward data...', 10, height / 2);
        return;
    }
    
    const padding = { top: 25, right: 10, bottom: 25, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;
    
    // Find min/max for scaling
    let minVal = Math.min(...data);
    let maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;
    minVal -= range * 0.1;
    maxVal += range * 0.1;
    
    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH * i / 4);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        // Y-axis labels
        const val = maxVal - (i / 4) * (maxVal - minVal);
        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(0), padding.left - 5, y + 3);
    }
    
    // Draw zero line if in range
    if (minVal < 0 && maxVal > 0) {
        const zeroY = padding.top + chartH - ((0 - minVal) / (maxVal - minVal) * chartH);
        ctx.strokeStyle = '#666';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, zeroY);
        ctx.lineTo(width - padding.right, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    // Draw reward line
    ctx.strokeStyle = '#4ade80';  // Green
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
        const x = padding.left + (i / (data.length - 1)) * chartW;
        const normalized = (data[i] - minVal) / (maxVal - minVal);
        const y = padding.top + chartH - (normalized * chartH);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // X-axis label
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`Last ${data.length} PPO updates`, width / 2, height - 5);
    
    // Current value at top
    const latest = data[data.length - 1];
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#4ade80';
    ctx.fillText(`Avg Reward: ${latest.toFixed(1)}`, padding.left, padding.top - 8);
}
