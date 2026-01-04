// Catmull-Rom spline interpolation
function catmullRom(p0, p1, p2, p3, t) {
    const t2 = t * t;
    const t3 = t2 * t;
    
    return {
        x: 0.5 * (
            (2 * p1.x) +
            (-p0.x + p2.x) * t +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
            (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
        ),
        y: 0.5 * (
            (2 * p1.y) +
            (-p0.y + p2.y) * t +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
            (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
        )
    };
}

// Catmull-Rom derivative (tangent, not normalized)
function catmullRomDerivative(p0, p1, p2, p3, t) {
    const t2 = t * t;
    
    return {
        x: 0.5 * (
            (-p0.x + p2.x) +
            2 * (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t +
            3 * (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t2
        ),
        y: 0.5 * (
            (-p0.y + p2.y) +
            2 * (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t +
            3 * (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t2
        )
    };
}

export class ClosedSpline {
    #points;
    #segmentCount;
    
    constructor(controlPoints) {
        if (controlPoints.length < 3) {
            throw new Error('ClosedSpline requires at least 3 control points');
        }
        this.#points = controlPoints;
        this.#segmentCount = controlPoints.length;
    }
    
    // Get the 4 control points for a segment, with wrap-around
    #getSegmentPoints(segmentIndex) {
        const n = this.#segmentCount;
        const i = ((segmentIndex % n) + n) % n;
        
        return {
            p0: this.#points[(i - 1 + n) % n],
            p1: this.#points[i],
            p2: this.#points[(i + 1) % n],
            p3: this.#points[(i + 2) % n]
        };
    }
    
    // Sample point at t ∈ [0, 1), wraps seamlessly
    sample(t) {
        // Normalize t to [0, 1)
        t = ((t % 1) + 1) % 1;
        
        const segment = Math.floor(t * this.#segmentCount);
        const localT = (t * this.#segmentCount) % 1;
        
        const { p0, p1, p2, p3 } = this.#getSegmentPoints(segment);
        return catmullRom(p0, p1, p2, p3, localT);
    }
    
    // Tangent at t (normalized)
    tangent(t) {
        t = ((t % 1) + 1) % 1;
        
        const segment = Math.floor(t * this.#segmentCount);
        const localT = (t * this.#segmentCount) % 1;
        
        const { p0, p1, p2, p3 } = this.#getSegmentPoints(segment);
        const deriv = catmullRomDerivative(p0, p1, p2, p3, localT);
        
        const len = Math.hypot(deriv.x, deriv.y);
        if (len === 0) return { x: 1, y: 0 };
        return { x: deriv.x / len, y: deriv.y / len };
    }
    
    // Normal at t (perpendicular to tangent, pointing "left")
    normal(t) {
        const tan = this.tangent(t);
        return { x: -tan.y, y: tan.x };
    }
    
    get pointCount() {
        return this.#points.length;
    }
    
    // Get raw control points (for visualization)
    getControlPoints() {
        return [...this.#points];
    }
}

