const elements = {
    gen: document.getElementById('gen-count'),
    best: document.getElementById('best-fit'),
    alive: document.getElementById('ai-alive'),
    total: document.getElementById('ai-total'),
    msg: document.getElementById('status-msg'),
    actorStats: document.getElementById('actor-stats'),
    criticStats: document.getElementById('critic-stats'),
    avgReward: document.getElementById('avg-reward'),
    progress: document.getElementById('leader-progress'),
};

export function updateUI({ generation, bestFitness, aliveCount, totalCount, message, actorStats, criticStats, avgReward, leaderProgress }) {
    if (elements.gen) elements.gen.textContent = generation;
    if (elements.best) elements.best.textContent = Math.floor(bestFitness);
    if (elements.alive) elements.alive.textContent = aliveCount;
    if (elements.total) elements.total.textContent = totalCount;
    if (elements.msg) elements.msg.textContent = message;
    if (elements.actorStats) elements.actorStats.textContent = actorStats;
    if (elements.criticStats) elements.criticStats.textContent = criticStats;
    if (elements.avgReward) elements.avgReward.textContent = avgReward.toFixed(1);
    if (elements.progress) elements.progress.textContent = Math.floor(leaderProgress * 100) + '%';
}
