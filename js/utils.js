// Line segment intersection: returns {x, y, offset} or null
export function getIntersection(A, B, C, D) {
    const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
    const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
    const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);
    
    if (bottom !== 0) {
        const t = tTop / bottom;
        const u = uTop / bottom;
        if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
            return {
                x: A.x + t * (B.x - A.x),
                y: A.y + t * (B.y - A.y),
                offset: t
            };
        }
    }
    return null;
}

// Distance from point p to line segment v-w
export function distToSegment(p, v, w) {
    const l2 = (v.x - w.x) ** 2 + (v.y - w.y) ** 2;
    if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
    
    const t = Math.max(0, Math.min(1, 
        ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2
    ));
    
    return Math.hypot(
        p.x - (v.x + t * (w.x - v.x)),
        p.y - (v.y + t * (w.y - v.y))
    );
}

// Ray-circle intersection: returns distance or null
export function rayCircleIntersection(rayOrigin, rayDir, circleCenter, radius) {
    const L = {
        x: circleCenter.x - rayOrigin.x,
        y: circleCenter.y - rayOrigin.y
    };
    const tca = L.x * rayDir.x + L.y * rayDir.y;
    if (tca < 0) return null;
    
    const d2 = (L.x * L.x + L.y * L.y) - tca * tca;
    if (d2 > radius * radius) return null;
    
    const thc = Math.sqrt(radius * radius - d2);
    let t0 = tca - thc;
    if (t0 < 0) t0 = tca + thc;
    if (t0 < 0) return null;
    
    return t0;
}

// Normalize a vector
export function normalize(v) {
    const len = Math.hypot(v.x, v.y);
    if (len === 0) return { x: 0, y: 0 };
    return { x: v.x / len, y: v.y / len };
}

// Perpendicular vector (90° counterclockwise)
export function perpendicular(v) {
    return { x: -v.y, y: v.x };
}

// Linear interpolation
export function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Calculate average of an array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Average, or 0 if array is empty
 */
export function average(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/**
 * Push item to array with maximum length limit (rolling window)
 * @param {Array} arr - Array to push to (mutated)
 * @param {*} item - Item to push
 * @param {number} maxLength - Maximum array length
 */
export function pushWithLimit(arr, item, maxLength) {
    arr.push(item);
    if (arr.length > maxLength) {
        arr.shift();
    }
}

/**
 * Normalize angle to [-π, π] range
 * @param {number} angle - Angle in radians
 * @returns {number} Normalized angle in [-π, π]
 */
export function normalizeAngle(angle) {
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle < -Math.PI) angle += 2 * Math.PI;
    return angle;
}

/**
 * Calculate grid spawn position for a car
 * @param {number} id - Car ID (determines grid slot)
 * @param {Object} startLine - { point, normal, tangent } from track.getStartLine()
 * @param {Object} gridConfig - { lateralOffset, rowSpacing, startOffset }
 * @returns {{ x: number, y: number, angle: number }}
 */
export function getGridPosition(id, startLine, gridConfig) {
    const col = (id % 2 === 0) ? -1 : 1;
    const row = Math.floor(id / 2);
    
    const x = startLine.point.x 
        + startLine.normal.x * col * gridConfig.lateralOffset
        - startLine.tangent.x * (row * gridConfig.rowSpacing + gridConfig.startOffset);
    const y = startLine.point.y 
        + startLine.normal.y * col * gridConfig.lateralOffset
        - startLine.tangent.y * (row * gridConfig.rowSpacing + gridConfig.startOffset);
    const angle = Math.atan2(startLine.tangent.y, startLine.tangent.x);
    
    return { x, y, angle };
}

