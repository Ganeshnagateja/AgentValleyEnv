import { useState, useCallback, useRef, useEffect } from 'react';
import type { BackendTrainingMetric, GameState, TrainingMode, TrainingStartOptions } from '@/types/game';
import { AgentEngine } from '@/game/AgentEngine';
import { EnvironmentEngine } from '@/game/Environment';
import { apiClient } from '@/lib/apiClient';

const createInitialAgents = () => [
  AgentEngine.createAgent('farmer', 0),
  AgentEngine.createAgent('farmer', 1),
  AgentEngine.createAgent('miner', 2),
  AgentEngine.createAgent('miner', 3),
  AgentEngine.createAgent('builder', 4),
  AgentEngine.createAgent('builder', 5),
  AgentEngine.createAgent('warrior', 6),
  AgentEngine.createAgent('warrior', 7),
];

const INITIAL_TRAINING = {
  isTraining: false,
  episode: 0,
  maxEpisodes: 0,
  mode: 'grpo' as TrainingMode,
  difficulty: 'mixed',
  metrics: [] as BackendTrainingMetric[],
  currentMetrics: null,
  bestReward: -Infinity,
  avgReward: 0,
  loss: 0,
  learningCurve: [] as Array<{ episode: number; reward: number; loss: number | null }>,
  realEnvResult: null,
  backendOnline: false,
  taskScore: 0,
  finalStatus: 'unknown',
  policyLoss: null,
  klDivergence: null,
  entropy: null,
  epsilon: null,
  checkpointPath: null,
  metricsPath: null,
  lastUpdated: null,
  trainingError: null,
};

function summarizeMetrics(metrics: BackendTrainingMetric[]) {
  const rewards = metrics.map(m => m.total_reward);
  const bestReward = rewards.length ? Math.max(...rewards) : -Infinity;
  const avgReward = rewards.length ? rewards.reduce((a, b) => a + b, 0) / rewards.length : 0;
  return { bestReward, avgReward };
}

function resultFromMetric(metric: BackendTrainingMetric | null, backendOnline: boolean) {
  if (!metric) return null;
  return {
    episode: metric.episode,
    difficulty: metric.difficulty,
    totalReward: metric.total_reward,
    finalStatus: metric.final_status,
    taskScore: metric.task_score,
    totalSteps: metric.steps,
    backendOnline,
  };
}

export function useGameLoop() {
  const [gameState, setGameState] = useState<GameState>({
    tick: 0,
    isRunning: false,
    speed: 1,
    agents: createInitialAgents(),
    resources: [],
    events: [],
    market: {
      prices: { food: 10, wood: 15, stone: 20, gold: 50, ore: 35 },
      history: [],
      volatility: 0.1,
    },
    training: INITIAL_TRAINING,
    selectedAgent: null,
    view: 'title',
  });

  const envRef = useRef(new EnvironmentEngine());
  const tickRef = useRef(0);
  const animFrameRef = useRef<number>(0);
  const gameStateRef = useRef(gameState);

  useEffect(() => { gameStateRef.current = gameState; }, [gameState]);

  useEffect(() => {
    setGameState(prev => ({ ...prev, resources: envRef.current.resources }));
  }, []);

  const syncBackendTraining = useCallback(async (modeHint?: TrainingMode | null) => {
    const status = await apiClient.trainingStatus();
    const health = await apiClient.health();
    const mode = (status?.mode ?? modeHint ?? gameStateRef.current.training.mode) as TrainingMode;
    const metricPayload = await apiClient.trainingMetrics(mode, 300);
    const latestPayload = await apiClient.trainingLatest(mode);
    const metrics = metricPayload?.metrics ?? [];
    const latest = status?.latest_metric ?? latestPayload?.latest_metric ?? metrics[metrics.length - 1] ?? null;
    const backendOnline = health?.status === 'ok';
    const summary = summarizeMetrics(metrics);

    setGameState(prev => ({
      ...prev,
      training: {
        ...prev.training,
        backendOnline,
        isTraining: Boolean(status?.running),
        mode,
        episode: latest?.episode ?? status?.episode ?? 0,
        maxEpisodes: status?.episodes_requested ?? prev.training.maxEpisodes,
        difficulty: latest?.difficulty ?? status?.difficulty ?? prev.training.difficulty,
        metrics,
        currentMetrics: latest,
        bestReward: summary.bestReward,
        avgReward: summary.avgReward,
        loss: latest?.policy_loss ?? 0,
        learningCurve: metrics.map(m => ({
          episode: m.episode,
          reward: m.total_reward,
          loss: m.policy_loss ?? null,
        })),
        realEnvResult: resultFromMetric(latest, backendOnline),
        taskScore: latest?.task_score ?? 0,
        finalStatus: latest?.final_status ?? 'unknown',
        policyLoss: latest?.policy_loss ?? null,
        klDivergence: latest?.kl_divergence ?? null,
        entropy: latest?.entropy ?? null,
        epsilon: latest?.epsilon ?? null,
        checkpointPath: latest?.checkpoint_path ?? status?.checkpoint_path ?? null,
        metricsPath: metricPayload?.metrics_path ?? status?.metrics_path ?? null,
        lastUpdated: latest?.updated_at ?? status?.updated_at ?? null,
        trainingError: status?.error ?? null,
      },
    }));
  }, []);

  useEffect(() => {
    syncBackendTraining();
    const id = window.setInterval(() => {
      const current = gameStateRef.current;
      if (current.training.isTraining || current.view === 'training') {
        syncBackendTraining();
      } else {
        apiClient.health().then(res => {
          setGameState(prev => ({
            ...prev,
            training: { ...prev.training, backendOnline: res?.status === 'ok' },
          }));
        });
      }
    }, 1500);
    return () => window.clearInterval(id);
  }, [syncBackendTraining]);

  const processTick = useCallback(() => {
    setGameState(prev => {
      if (!prev.isRunning) return prev;

      const env = envRef.current;
      tickRef.current++;

      const updatedAgents = prev.agents.map(agent => {
        if (agent.energy <= 0) {
          agent.state = 'resting';
          agent.energy = Math.min(100, agent.energy + 20);
          return { ...agent };
        }
        if (tickRef.current % 10 === 0 || agent.state === 'idle') {
          const { agent: updated } = AgentEngine.runOODA(agent, prev.agents, tickRef.current);
          agent = updated;
        }
        if (agent.ooda.act.executing) {
          agent = AgentEngine.executeAction(agent, agent.ooda.decide.chosenAction);
        }
        if (agent.state === 'moving' || agent.state === 'idle') {
          agent.position.x += (Math.random() - 0.5) * 4;
          agent.position.y += (Math.random() - 0.5) * 4;
          agent.position.x = Math.max(30, Math.min(770, agent.position.x));
          agent.position.y = Math.max(70, Math.min(490, agent.position.y));
        }
        agent.energy = Math.max(0, agent.energy - 0.1);
        return { ...agent };
      });

      const { event, market } = env.tickUpdate(updatedAgents);
      const newEvents = event ? [...prev.events, event] : prev.events;
      if (newEvents.length > 20) newEvents.shift();

      return {
        ...prev,
        tick: tickRef.current,
        agents: updatedAgents,
        resources: [...env.resources],
        events: newEvents,
        market,
      };
    });
  }, []);

  useEffect(() => {
    let lastTime = 0;
    const interval = 1000 / (20 * gameState.speed);
    const loop = (time: number) => {
      if (time - lastTime >= interval) {
        processTick();
        lastTime = time;
      }
      animFrameRef.current = requestAnimationFrame(loop);
    };
    if (gameState.isRunning) animFrameRef.current = requestAnimationFrame(loop);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [gameState.isRunning, gameState.speed, processTick]);

  const pauseGame = useCallback(() => setGameState(prev => ({ ...prev, isRunning: false })), []);
  const resumeGame = useCallback(() => setGameState(prev => ({ ...prev, isRunning: true })), []);
  const setSpeed = useCallback((speed: number) => setGameState(prev => ({ ...prev, speed })), []);
  const selectAgent = useCallback((id: string | null) => setGameState(prev => ({ ...prev, selectedAgent: id })), []);
  const setView = useCallback((view: 'title' | 'game' | 'training' | 'dashboard') => setGameState(prev => ({ ...prev, view })), []);

  const startTraining = useCallback(async (options: TrainingStartOptions) => {
    setGameState(prev => ({
      ...prev,
      training: {
        ...prev.training,
        isTraining: true,
        mode: options.mode,
        maxEpisodes: options.episodes,
        difficulty: options.difficulty,
        metrics: [],
        currentMetrics: null,
        learningCurve: [],
        realEnvResult: null,
        trainingError: null,
      },
    }));
    const response = await apiClient.startTraining({
      mode: options.mode,
      episodes: options.episodes,
      difficulty: options.difficulty,
      seed: options.seed,
      reset_metrics: true,
      group_size: options.mode === 'grpo' || options.mode === 'multi_agent_grpo' ? 4 : undefined,
    });
    if (!response) {
      setGameState(prev => ({
        ...prev,
        training: { ...prev.training, isTraining: false, backendOnline: false, trainingError: 'Backend did not accept training start.' },
      }));
      return;
    }
    await syncBackendTraining(options.mode);
  }, [syncBackendTraining]);

  const stopTraining = useCallback(async () => {
    await apiClient.stopTraining();
    await syncBackendTraining();
  }, [syncBackendTraining]);

  const resetSimulation = useCallback(() => {
    envRef.current = new EnvironmentEngine();
    tickRef.current = 0;
    setGameState(prev => ({
      ...prev,
      tick: 0,
      agents: createInitialAgents(),
      isRunning: false,
      resources: envRef.current.resources,
      events: [],
      selectedAgent: null,
    }));
  }, []);

  const enterTheValley = useCallback(() => setGameState(prev => ({ ...prev, view: 'game', isRunning: true })), []);

  return {
    gameState,
    startGame: enterTheValley,
    pauseGame,
    resumeGame,
    setSpeed,
    selectAgent,
    setView,
    startTraining,
    stopTraining,
    resetSimulation,
    enterTheValley,
  };
}
