import { CONFIG } from './config.js';
import { ClosedSpline } from './spline.js';

export class Track {
    #spline;
    #halfWidth;
    #samples;
    
    constructor(controlPoints, halfWidth = CONFIG.ROAD_HALF_WIDTH) {
        this.#spline = new ClosedSpline(controlPoints);
        this.#halfWidth = halfWidth;
        this.#buildSamples(CONFIG.TRACK_SAMPLES);
    }
    
    // === SDF Functions ===
    
    distanceToRoad(x, y) {
        const closest = this.getClosestPoint(x, y);
        return closest.distance - this.#halfWidth;
    }
    
    isOnRoad(x, y) {
        return this.distanceToRoad(x, y) <= 0;
    }
    
    isOffRoad(x, y) {
        return this.distanceToRoad(x, y) > 0;
    }
    
    // === Progress Tracking ===
    
    getClosestPoint(x, y) {
        let best = { distance: Infinity, t: 0, point: null, tangent: null };
        
        for (const sample of this.#samples) {
            const d = Math.hypot(x - sample.point.x, y - sample.point.y);
            if (d < best.distance) {
                best = { 
                    distance: d, 
                    t: sample.t, 
                    point: sample.point,
                    tangent: sample.tangent  // Track direction at this point
                };
            }
        }
        
        return best;
    }
    
    getProgress(x, y) {
        return this.getClosestPoint(x, y).t;
    }
    
    // Get track direction (tangent) at position
    getTrackDirection(x, y) {
        return this.getClosestPoint(x, y).tangent;
    }
    
    // === For Sensors (SDF-based ray marching) ===
    
    // Returns distance along ray until hitting road edge (SDF > 0)
    raycastToEdge(originX, originY, dirX, dirY, maxDist) {
        const stepSize = 8;  // Coarse steps
        let dist = 0;
        
        // March along the ray
        while (dist < maxDist) {
            const x = originX + dirX * dist;
            const y = originY + dirY * dist;
            const sdf = this.distanceToRoad(x, y);
            
            // Off road - we've hit the edge
            if (sdf > 0) {
                // Refine with smaller steps
                dist -= stepSize;
                for (let fine = 0; fine < stepSize; fine += 2) {
                    const fx = originX + dirX * (dist + fine);
                    const fy = originY + dirY * (dist + fine);
                    if (this.distanceToRoad(fx, fy) > 0) {
                        return dist + fine;
                    }
                }
                return dist + stepSize;
            }
            
            dist += stepSize;
        }
        
        return maxDist;
    }
    
    // === Start Grid ===
    
    getStartLine() {
        const point = this.#spline.sample(0);
        const tangent = this.#spline.tangent(0);
        const normal = this.#spline.normal(0);
        return { point, tangent, normal };
    }
    
    // === Rendering ===
    
    #getEdgePaths() {
        const innerPath = [];
        const outerPath = [];
        for (const sample of this.#samples) {
            innerPath.push({
                x: sample.point.x - sample.normal.x * this.#halfWidth,
                y: sample.point.y - sample.normal.y * this.#halfWidth
            });
            outerPath.push({
                x: sample.point.x + sample.normal.x * this.#halfWidth,
                y: sample.point.y + sample.normal.y * this.#halfWidth
            });
        }
        return { innerPath, outerPath };
    }
    
    #drawPath(ctx, path, close = true) {
        ctx.beginPath();
        path.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        if (close) ctx.closePath();
    }
    
    #drawRoadSurface(ctx, innerPath, outerPath) {
        ctx.fillStyle = '#1a2a1a';
        this.#drawPath(ctx, innerPath);
        for (let i = outerPath.length - 1; i >= 0; i--) {
            ctx.lineTo(outerPath[i].x, outerPath[i].y);
        }
        ctx.closePath();
        ctx.fill();
    }
    
    #drawRoadEdges(ctx, innerPath, outerPath) {
        ctx.strokeStyle = '#225e32';
        ctx.lineWidth = 14;
        ctx.lineJoin = 'round';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#225e32';
        
        this.#drawPath(ctx, innerPath);
        ctx.stroke();
        this.#drawPath(ctx, outerPath);
        ctx.stroke();
        
        ctx.shadowBlur = 0;
    }
    
    #drawStartLine(ctx) {
        const start = this.getStartLine();
        ctx.strokeStyle = '#c41e3a';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(
            start.point.x - start.normal.x * this.#halfWidth,
            start.point.y - start.normal.y * this.#halfWidth
        );
        ctx.lineTo(
            start.point.x + start.normal.x * this.#halfWidth,
            start.point.y + start.normal.y * this.#halfWidth
        );
        ctx.stroke();
    }
    
    #drawCenterline(ctx) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 2;
        ctx.setLineDash([20, 20]);
        this.#drawPath(ctx, this.#samples.map(s => s.point));
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    draw(ctx) {
        const { innerPath, outerPath } = this.#getEdgePaths();
        this.#drawRoadSurface(ctx, innerPath, outerPath);
        this.#drawRoadEdges(ctx, innerPath, outerPath);
        this.#drawStartLine(ctx);
        this.#drawCenterline(ctx);
    }
    
    // === For testing/debugging ===
    
    getSamples() {
        return this.#samples;
    }
    
    getHalfWidth() {
        return this.#halfWidth;
    }
    
    // === Private ===
    
    #buildSamples(resolution) {
        this.#samples = [];
        for (let i = 0; i < resolution; i++) {
            const t = i / resolution;
            this.#samples.push({
                t,
                point: this.#spline.sample(t),
                tangent: this.#spline.tangent(t),
                normal: this.#spline.normal(t)
            });
        }
    }
}

