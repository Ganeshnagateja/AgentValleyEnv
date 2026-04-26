import { useEffect, useRef, useState } from 'react';
import type { Agent, BackendTrainingMetric, TrainingMode, TrainingStartOptions, TrainingState } from '@/types/game';
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  CheckCircle,
  Cpu,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  TrendingUp,
  Users,
  XCircle,
  Zap,
} from 'lucide-react';

interface TrainingDashboardProps {
  training: TrainingState;
  agents: Agent[];
  onStartTraining: (options: TrainingStartOptions) => void;
  onStopTraining: () => void;
  onReset: () => void;
  onBack: () => void;
}

type Point = { episode: number; value: number };

const MODE_LABELS: Record<TrainingMode, string> = {
  q_learning: 'Q-learning',
  neural_policy: 'Neural policy',
  grpo: 'GRPO',
  multi_agent_grpo: 'Multi-agent GRPO',
};

function numberOrDash(value: number | null | undefined, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function percentOrDash(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'N/A';
}

function metricPoint(metrics: BackendTrainingMetric[], key: keyof BackendTrainingMetric): Point[] {
  return metrics
    .filter(metric => typeof metric[key] === 'number')
    .map(metric => ({ episode: metric.episode, value: metric[key] as number }));
}

function LineChart({ data, color, label, height = 150 }: {
  data: Point[];
  color: string;
  label: string;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth * 2;
    canvas.height = height * 2;
    ctx.scale(2, 2);

    const w = canvas.offsetWidth;
    const h = height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {
      const y = (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (data.length < 2) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = '12px sans-serif';
      ctx.fillText('Waiting for backend metrics', 10, 22);
      return;
    }

    const values = data.map(d => d.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const pad = Math.max((maxVal - minVal) * 0.12, 0.01);
    const low = minVal - pad;
    const high = maxVal + pad;
    const range = high - low || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    data.forEach((d, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((d.value - low) / range) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const last = data[data.length - 1];
    ctx.fillStyle = color;
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText(`${label}: ${last.value.toFixed(3)}`, 8, 14);
  }, [data, color, label, height]);

  return (
    <div className="bg-white/5 rounded-lg p-3 border border-white/10">
      <canvas ref={canvasRef} className="w-full" style={{ height }} />
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: {
  icon: any;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-white/5 rounded-lg p-3 border border-white/10">
      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-4 h-4" style={{ color }} />
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <div className="text-lg font-bold text-white truncate" title={value}>{value}</div>
    </div>
  );
}

export default function TrainingDashboard({
  training,
  agents,
  onStartTraining,
  onStopTraining,
  onReset,
  onBack,
}: TrainingDashboardProps) {
  const [mode, setMode] = useState<TrainingMode>(training.mode || 'grpo');
  const [episodes, setEpisodes] = useState(10);
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard' | 'mixed'>('mixed');
  const [seed, setSeed] = useState(42);

  const latest = training.currentMetrics;
  const rewardData = metricPoint(training.metrics, 'total_reward');
  const scoreData = metricPoint(training.metrics, 'task_score');
  const lossData = metricPoint(training.metrics, 'policy_loss');
  const meanQData = metricPoint(training.metrics, 'mean_q_value');
  const maxQData = metricPoint(training.metrics, 'max_q_value');
  const epsilonData = metricPoint(training.metrics, 'epsilon');
  const groupRewardData = metricPoint(training.metrics, 'mean_group_reward');
  const teamRewardData = metricPoint(training.metrics, 'total_team_reward');
  const cooperationRateData = metricPoint(training.metrics, 'cooperation_rate');
  const klData = metricPoint(training.metrics, 'kl_divergence');
  const entropyData = metricPoint(training.metrics, 'entropy');

  const progress = training.maxEpisodes > 0 ? (training.episode / training.maxEpisodes) * 100 : 0;
  const statusIcon = training.backendOnline ? CheckCircle : XCircle;
  const StatusIcon = statusIcon;

  return (
    <div className="screen-scroll bg-black text-white">
      <div className="border-b border-white/10 bg-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 screen-scroll-content">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4 min-w-0">
              <button onClick={onBack} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="min-w-0">
                <h1 className="text-xl font-bold flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-amber-400" />
                  Backend RL Training
                </h1>
                <p className="text-xs text-gray-500">Charts read JSONL metrics from the Python FastAPI backend.</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {!training.isTraining ? (
                <button
                  onClick={() => onStartTraining({ mode, episodes, difficulty, seed })}
                  className="px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg font-medium text-sm transition-colors flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Start
                </button>
              ) : (
                <button
                  onClick={onStopTraining}
                  className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium text-sm transition-colors flex items-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  Stop
                </button>
              )}
              <button onClick={onReset} className="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Reset visual simulation">
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-3">
            <div>
              <label className="text-[10px] text-gray-500 uppercase tracking-widest">Mode</label>
              <select
                value={mode}
                onChange={event => setMode(event.target.value as TrainingMode)}
                className="mt-1 w-full bg-black border border-white/15 rounded-lg px-3 py-2 text-sm text-white"
                disabled={training.isTraining}
              >
                <option value="q_learning">Q-learning</option>
                <option value="neural_policy">Neural policy</option>
                <option value="grpo">GRPO</option>
                <option value="multi_agent_grpo">Multi-agent GRPO</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase tracking-widest">Difficulty</label>
              <select
                value={difficulty}
                onChange={event => setDifficulty(event.target.value as typeof difficulty)}
                className="mt-1 w-full bg-black border border-white/15 rounded-lg px-3 py-2 text-sm text-white"
                disabled={training.isTraining}
              >
                <option value="mixed">Mixed curriculum</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase tracking-widest">Episodes</label>
              <input
                type="number"
                min={1}
                max={200}
                value={episodes}
                onChange={event => setEpisodes(Number(event.target.value))}
                className="mt-1 w-full bg-black border border-white/15 rounded-lg px-3 py-2 text-sm text-white"
                disabled={training.isTraining}
              />
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase tracking-widest">Seed</label>
              <input
                type="number"
                value={seed}
                onChange={event => setSeed(Number(event.target.value))}
                className="mt-1 w-full bg-black border border-white/15 rounded-lg px-3 py-2 text-sm text-white"
                disabled={training.isTraining}
              />
            </div>
          </div>

          {training.isTraining && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-400">Episode {training.episode} / {training.maxEpisodes}</span>
                <span className="text-xs text-amber-400">{Math.min(100, progress).toFixed(1)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-amber-500 rounded-full transition-all" style={{ width: `${Math.min(100, progress)}%` }} />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-4 screen-scroll-content">
        {training.trainingError && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {training.trainingError}
          </div>
        )}

        <div className="mb-4 flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-4 py-3">
          <div className="flex items-center gap-2 text-sm">
            <StatusIcon className={`w-4 h-4 ${training.backendOnline ? 'text-green-400' : 'text-red-400'}`} />
            <span className={training.backendOnline ? 'text-green-300' : 'text-red-300'}>
              {training.backendOnline ? 'Backend online' : 'Backend offline'}
            </span>
          </div>
          <div className="text-xs text-gray-500">
            Visual simulation agents: <span className="text-gray-300">{agents.length}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
          <StatCard icon={BrainCircuit} label="Training Mode" value={MODE_LABELS[training.mode]} color="#A78BFA" />
          <StatCard icon={Activity} label="Episode" value={`${training.episode}`} color="#34D399" />
          <StatCard icon={ShieldCheck} label="Difficulty" value={latest?.difficulty ?? training.difficulty} color="#FBBF24" />
          <StatCard icon={TrendingUp} label="Total Reward" value={numberOrDash(latest?.total_reward, 3)} color="#FBBF24" />
          <StatCard icon={ShieldCheck} label="Task Score" value={percentOrDash(latest?.task_score)} color="#60A5FA" />
          <StatCard icon={Zap} label="Final Status" value={latest?.final_status ?? 'N/A'} color="#34D399" />
          <StatCard icon={Activity} label="Policy Loss" value={numberOrDash(latest?.policy_loss, 6)} color="#F87171" />
          <StatCard icon={Activity} label="KL Divergence" value={numberOrDash(latest?.kl_divergence, 6)} color="#FB923C" />
          <StatCard icon={Activity} label="Entropy" value={numberOrDash(latest?.entropy, 4)} color="#A78BFA" />
          <StatCard icon={Activity} label="Epsilon" value={numberOrDash(latest?.epsilon, 4)} color="#38BDF8" />
          <StatCard icon={Users} label="Cooperation" value={percentOrDash(latest?.cooperation_rate)} color="#34D399" />
          <StatCard icon={Cpu} label="Checkpoint" value={training.checkpointPath ?? 'N/A'} color="#34D399" />
          <StatCard icon={Cpu} label="Last Updated" value={training.lastUpdated ?? 'N/A'} color="#9CA3AF" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <div>
            <h3 className="text-sm text-gray-400 mb-2 flex items-center gap-1">
              <TrendingUp className="w-3 h-3" /> Total Reward per Episode
            </h3>
            <LineChart data={rewardData} color="#FBBF24" label="Reward" />
          </div>
          <div>
            <h3 className="text-sm text-gray-400 mb-2 flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" /> Backend Task Score
            </h3>
            <LineChart data={scoreData} color="#60A5FA" label="Task score" />
          </div>
        </div>

        {training.mode === 'q_learning' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <LineChart data={meanQData} color="#34D399" label="Mean Q" />
            <LineChart data={maxQData} color="#A78BFA" label="Max Q" />
            <LineChart data={epsilonData} color="#38BDF8" label="Epsilon" />
          </div>
        )}

        {training.mode === 'neural_policy' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <LineChart data={lossData} color="#F87171" label="Policy loss" />
            <LineChart data={entropyData} color="#A78BFA" label="Entropy" />
          </div>
        )}

        {training.mode === 'grpo' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <LineChart data={lossData} color="#F87171" label="Policy loss" />
            <LineChart data={groupRewardData} color="#34D399" label="Mean group reward" />
            <LineChart data={klData} color="#FB923C" label="KL" />
            <LineChart data={entropyData} color="#A78BFA" label="Entropy" />
          </div>
        )}

        {training.mode === 'multi_agent_grpo' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <LineChart data={teamRewardData.length ? teamRewardData : rewardData} color="#FBBF24" label="Team reward" />
            <LineChart data={cooperationRateData} color="#34D399" label="Cooperation rate" />
            <LineChart data={lossData} color="#F87171" label="Policy loss" />
            <LineChart data={groupRewardData} color="#60A5FA" label="Mean group reward" />
            <LineChart data={entropyData} color="#A78BFA" label="Entropy" />
          </div>
        )}

        <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-4">
          <h3 className="text-sm font-bold text-amber-300 mb-2">Visual Simulation Is Separate</h3>
          <p className="text-xs text-amber-100/70">
            The animated valley still uses local movement and OODA visuals. The learning curves above use only backend JSONL metrics from real environment rollouts.
          </p>
        </div>
      </div>
    </div>
  );
}
