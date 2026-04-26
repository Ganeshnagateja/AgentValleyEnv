// Legacy Agent Valley visual-demo server.
// The active backend is the Python FastAPI/OpenEnv server in ../../server/app.py.
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// In-memory state store (would be a database in production)
const state = {
  episodes: [],
  agents: [],
  metrics: [],
  environment: {
    regions: [
      { id: 'farmland', name: 'Green Meadows', type: 'farmland', danger: 0.1 },
      { id: 'mine', name: 'Crystal Mines', type: 'mine', danger: 0.4 },
      { id: 'forest', name: 'Whispering Woods', type: 'forest', danger: 0.1 },
      { id: 'village', name: 'Trader\'s Post', type: 'village', danger: 0.1 },
      { id: 'wilderness', name: 'Wild Frontier', type: 'wilderness', danger: 0.7 },
    ],
    market: {
      prices: { food: 10, wood: 15, stone: 20, gold: 50, ore: 35 },
      volatility: 0.1,
    },
  },
  config: {
    maxEpisodes: 1000,
    agentCount: 8,
    learningRate: 0.1,
    explorationRate: 0.3,
    cooperationBias: 0.5,
  },
};

// ============ API ROUTES ============

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Get environment configuration
app.get('/api/environment', (req, res) => {
  res.json(state.environment);
});

// Update environment configuration
app.post('/api/environment', (req, res) => {
  state.environment = { ...state.environment, ...req.body };
  res.json(state.environment);
});

// Get all agents
app.get('/api/agents', (req, res) => {
  res.json(state.agents);
});

// Update agents
app.post('/api/agents', (req, res) => {
  state.agents = req.body;
  res.json({ status: 'ok', count: state.agents.length });
});

// Get training config
app.get('/api/config', (req, res) => {
  res.json(state.config);
});

// Update training config
app.post('/api/config', (req, res) => {
  state.config = { ...state.config, ...req.body };
  res.json(state.config);
});

// Get training metrics
app.get('/api/metrics', (req, res) => {
  res.json(state.metrics);
});

// Add training metrics
app.post('/api/metrics', (req, res) => {
  const metric = { ...req.body, timestamp: Date.now() };
  state.metrics.push(metric);
  if (state.metrics.length > 10000) {
    state.metrics = state.metrics.slice(-5000);
  }
  res.json({ status: 'ok' });
});

// Get episodes
app.get('/api/episodes', (req, res) => {
  res.json(state.episodes);
});

// Add episode
app.post('/api/episodes', (req, res) => {
  const episode = { ...req.body, timestamp: Date.now() };
  state.episodes.push(episode);
  if (state.episodes.length > 5000) {
    state.episodes = state.episodes.slice(-2500);
  }
  res.json({ status: 'ok' });
});

// Get market prices
app.get('/api/market', (req, res) => {
  res.json(state.environment.market);
});

// Simulate market update
app.post('/api/market/update', (req, res) => {
  const { prices, volatility } = state.environment.market;
  for (const key of Object.keys(prices)) {
    const change = (Math.random() - 0.5) * 2 * volatility * prices[key];
    prices[key] = Math.max(1, Math.round(prices[key] + change));
  }
  res.json(state.environment.market);
});

// Get training summary
app.get('/api/training/summary', (req, res) => {
  if (state.metrics.length === 0) {
    return res.json({ message: 'No training data yet' });
  }
  const rewards = state.metrics.map(m => m.totalReward || 0);
  res.json({
    totalEpisodes: state.episodes.length,
    avgReward: rewards.reduce((a, b) => a + b, 0) / rewards.length,
    bestReward: Math.max(...rewards),
    worstReward: Math.min(...rewards),
    totalMetrics: state.metrics.length,
  });
});

// Reset all state
app.post('/api/reset', (req, res) => {
  state.episodes = [];
  state.metrics = [];
  state.agents = [];
  res.json({ status: 'ok', message: 'All state reset' });
});

// ============ STATIC FILES ============

// Serve static files from dist folder
app.use(express.static(path.join(__dirname, '../dist')));

// Serve index.html for all other routes (SPA support)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../dist', 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log(`Agent Valley Server running on port ${PORT}`);
  console.log(`API endpoints available at http://localhost:${PORT}/api/`);
});

module.exports = app;
