// Agent Valley - Multi-Agent RL Environment Types

export type AgentRole = 'farmer' | 'miner' | 'builder' | 'warrior';
export type AgentState = 'idle' | 'working' | 'trading' | 'fighting' | 'resting' | 'moving' | 'cooperating';
export type ResourceType = 'food' | 'wood' | 'stone' | 'gold' | 'ore';
export type ActionType = 'gather' | 'trade' | 'build' | 'explore' | 'rest' | 'cooperate' | 'compete' | 'defend';
export type TrainingMode = 'q_learning' | 'neural_policy' | 'grpo' | 'multi_agent_grpo';

export interface Position {
  x: number;
  y: number;
}

export interface Resource {
  type: ResourceType;
  amount: number;
  position: Position;
}

export interface AgentMemory {
  observations: Observation[];
  successfulActions: ActionRecord[];
  failedActions: ActionRecord[];
  cooperations: CooperationRecord[];
  competitions: CompetitionRecord[];
}

export interface Observation {
  tick: number;
  what: string;
  where: Position;
  importance: number;
}

export interface ActionRecord {
  action: ActionType;
  reward: number;
  tick: number;
}

export interface CooperationRecord {
  partnerId: string;
  result: 'success' | 'failure';
  reward: number;
  tick: number;
}

export interface CompetitionRecord {
  opponentId: string;
  result: 'win' | 'lose';
  reward: number;
  tick: number;
}

export interface OODAState {
  observe: {
    nearbyAgents: Agent[];
    nearbyResources: Resource[];
    threats: Agent[];
    opportunities: string[];
  };
  orient: {
    currentNeed: ResourceType | 'safety' | 'wealth' | 'cooperation';
    threatLevel: number;
    opportunityScore: number;
    personalityBias: number; // 0 = cooperative, 1 = competitive
  };
  decide: {
    chosenAction: ActionType;
    target: Position | string | null;
    confidence: number;
  };
  act: {
    executing: boolean;
    progress: number;
    result: 'pending' | 'success' | 'failure';
  };
}

export interface Agent {
  id: string;
  name: string;
  role: AgentRole;
  state: AgentState;
  position: Position;
  resources: Record<ResourceType, number>;
  health: number;
  energy: number;
  experience: number;
  level: number;
  ooda: OODAState;
  memory: AgentMemory;
  decisionPolicy: DecisionPolicy;
  skills: Record<string, number>;
  avatar: string;
  color: string;
}

export interface DecisionPolicy {
  explorationRate: number;
  cooperationBias: number;
  riskTolerance: number;
  learningRate: number;
  qValues: Record<string, number[]>;
}

export interface EnvironmentEvent {
  type: 'resource_spawn' | 'disaster' | 'market_fluctuation' | 'quest' | 'invasion';
  description: string;
  position: Position;
  severity: number;
  tick: number;
}

export interface TrainingMetrics {
  episode: number;
  totalReward: number;
  avgDecisionTime: number;
  cooperationRate: number;
  resourceEfficiency: number;
  agentPerformance: Record<string, AgentMetrics>;
  timestamp: number;
}

export interface AgentMetrics {
  rewards: number[];
  avgReward: number;
  actions: Record<ActionType, number>;
  cooperations: number;
  competitions: number;
  resourcesGathered: number;
  level: number;
}

export interface GameState {
  tick: number;
  isRunning: boolean;
  speed: number;
  agents: Agent[];
  resources: Resource[];
  events: EnvironmentEvent[];
  market: MarketState;
  training: TrainingState;
  selectedAgent: string | null;
  view: 'title' | 'game' | 'training' | 'dashboard';
}

export interface MarketState {
  prices: Record<ResourceType, number>;
  history: Array<Record<ResourceType, number>>;
  volatility: number;
}

export interface RewardBreakdown {
  progress_reward: number;
  objective_bonus: number;
  focus_bonus: number;
  cooperation_bonus: number;
  safety_penalty: number;
  loop_penalty: number;
  format_penalty: number;
}

export interface RealEnvResult {
  episode: number;
  difficulty: string;
  totalReward: number;
  finalStatus: string;
  taskScore: number;
  totalSteps: number;
  backendOnline: boolean;
}

export interface BackendTrainingMetric {
  mode: TrainingMode;
  episode: number;
  update_step?: number;
  difficulty: string;
  total_reward: number;
  task_score: number;
  final_status: string;
  steps: number;
  epsilon?: number;
  mean_q_value?: number;
  max_q_value?: number;
  policy_loss?: number;
  total_team_reward?: number;
  cooperation_rate?: number;
  mean_group_reward?: number;
  reward_std?: number;
  kl_divergence?: number;
  entropy?: number;
  action_accuracy?: number | null;
  safety_score?: number | null;
  checkpoint_path?: string;
  metrics_path?: string;
  updated_at?: string;
}

export interface TrainingStartOptions {
  mode: TrainingMode;
  episodes: number;
  difficulty: 'easy' | 'medium' | 'hard' | 'mixed';
  seed: number;
}

export interface TrainingState {
  isTraining: boolean;
  episode: number;
  maxEpisodes: number;
  mode: TrainingMode;
  difficulty: string;
  metrics: BackendTrainingMetric[];
  currentMetrics: BackendTrainingMetric | null;
  bestReward: number;
  avgReward: number;
  loss: number;
  learningCurve: Array<{episode: number; reward: number; loss: number | null}>;
  realEnvResult: RealEnvResult | null;
  backendOnline: boolean;
  taskScore: number;
  finalStatus: string;
  policyLoss: number | null;
  klDivergence: number | null;
  entropy: number | null;
  epsilon: number | null;
  checkpointPath: string | null;
  metricsPath: string | null;
  lastUpdated: string | null;
  trainingError: string | null;
}

export interface ValleyRegion {
  id: string;
  name: string;
  type: 'farmland' | 'mine' | 'forest' | 'village' | 'wilderness' | 'plains' | 'river' | 'hills' | 'coast';
  position: Position;
  size: number;
  resources: Resource[];
  agents: string[];
  danger: number;
}
