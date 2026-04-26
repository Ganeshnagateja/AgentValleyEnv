// Agent Valley – typed client for the Python FastAPI backend
// All calls go through /api/* (proxied to localhost:7860 in dev,
// served directly by FastAPI in production).

export interface ResetRequest {
  difficulty?: 'easy' | 'medium' | 'hard';
  episode_index?: number;
  seed?: number;
}

export interface StepRequest {
  action: {
    primary_action: string;
    focus_resource: string;
    cooperation_mode: string;
    risk_posture: string;
    rationale?: string;
  };
}

export interface RewardBreakdown {
  progress_reward: number;
  objective_bonus: number;
  focus_bonus: number;
  cooperation_bonus: number;
  safety_penalty: number;
  loop_penalty: number;
  format_penalty: number;
  total?: number;
}

export interface StepResponse {
  observation: Record<string, unknown> | null;
  reward: number;
  done: boolean;
  info: {
    step?: number;
    max_steps?: number;
    reward_breakdown?: RewardBreakdown;
    repeat_count?: number;
    requires_reset?: boolean;
    message?: string;
    episode_result?: {
      final_status: string;
      task_score: number;
      cumulative_reward: number;
      total_steps: number;
    };
  };
}

export interface ResetResponse {
  observation: Record<string, unknown>;
  state: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
}

export type TrainingMode = 'q_learning' | 'neural_policy' | 'grpo' | 'multi_agent_grpo';

export interface TrainingStartRequest {
  mode: TrainingMode;
  episodes: number;
  difficulty: 'easy' | 'medium' | 'hard' | 'mixed';
  seed: number;
  reset_metrics?: boolean;
  learning_rate?: number;
  group_size?: number;
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

export interface TrainingStatusResponse {
  running: boolean;
  mode: TrainingMode | null;
  episode: number;
  episodes_requested: number;
  difficulty: string | null;
  seed: number | null;
  latest_metric: BackendTrainingMetric | null;
  error: string | null;
  started_at: string | null;
  updated_at: string | null;
  checkpoint_path: string | null;
  metrics_path: string | null;
}

export interface TrainingMetricsResponse {
  mode: TrainingMode;
  metrics: BackendTrainingMetric[];
  metrics_path: string;
  updated_at: string;
}

export interface TrainingLatestResponse {
  mode: TrainingMode;
  latest_metric: BackendTrainingMetric | null;
  metrics_path: string;
  updated_at: string;
}

const BASE = '/api';

async function post<T>(path: string, body: unknown): Promise<T | null> {
  try {
    const res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

async function get<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

export const apiClient = {
  health: () => get<HealthResponse>('/health'),
  reset: (req?: ResetRequest) => post<ResetResponse>('/reset', req ?? {}),
  step: (req: StepRequest) => post<StepResponse>('/step', req),
  state: () => get<Record<string, unknown>>('/state'),
  trainingModes: () => get<Record<string, unknown>>('/training/modes'),
  startTraining: (req: TrainingStartRequest) => post<TrainingStatusResponse>('/training/start', { reset_metrics: true, ...req }),
  stopTraining: () => post<TrainingStatusResponse>('/training/stop', {}),
  trainingStatus: () => get<TrainingStatusResponse>('/training/status'),
  trainingMetrics: (mode?: TrainingMode | null, limit = 300) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (mode) params.set('mode', mode);
    return get<TrainingMetricsResponse>(`/training/metrics?${params.toString()}`);
  },
  trainingLatest: (mode?: TrainingMode | null) => {
    const params = new URLSearchParams();
    if (mode) params.set('mode', mode);
    const query = params.toString();
    return get<TrainingLatestResponse>(`/training/latest${query ? `?${query}` : ''}`);
  },
};

// Map TypeScript ActionType → Python action schema
type TSAction = 'gather' | 'trade' | 'build' | 'explore' | 'rest' | 'cooperate' | 'compete' | 'defend';

const ACTION_MAP: Record<TSAction, StepRequest['action']> = {
  gather:    { primary_action: 'gather',    focus_resource: 'food',  cooperation_mode: 'solo',       risk_posture: 'cautious',  rationale: 'Gathering food and resources to stabilize supply' },
  trade:     { primary_action: 'trade',     focus_resource: 'gold',  cooperation_mode: 'share',      risk_posture: 'balanced',  rationale: 'Trading surplus for gold to fund operations' },
  build:     { primary_action: 'build',     focus_resource: 'stone', cooperation_mode: 'solo',       risk_posture: 'balanced',  rationale: 'Building infrastructure to improve resilience' },
  explore:   { primary_action: 'explore',   focus_resource: 'none',  cooperation_mode: 'solo',       risk_posture: 'balanced',  rationale: 'Exploring to discover new resource nodes' },
  rest:      { primary_action: 'rest',      focus_resource: 'none',  cooperation_mode: 'solo',       risk_posture: 'cautious',  rationale: 'Resting to recover energy and avoid exhaustion' },
  cooperate: { primary_action: 'cooperate', focus_resource: 'none',  cooperation_mode: 'coordinate', risk_posture: 'balanced',  rationale: 'Coordinating with teammates for collective benefit' },
  compete:   { primary_action: 'compete',   focus_resource: 'gold',  cooperation_mode: 'solo',       risk_posture: 'aggressive',rationale: 'Competing to capture scarce gold resources' },
  defend:    { primary_action: 'defend',    focus_resource: 'none',  cooperation_mode: 'protect',    risk_posture: 'cautious',  rationale: 'Defending the settlement against incoming threats' },
};

export function mapActionToPython(tsAction: string): StepRequest['action'] {
  return ACTION_MAP[tsAction as TSAction] ?? ACTION_MAP.gather;
}
