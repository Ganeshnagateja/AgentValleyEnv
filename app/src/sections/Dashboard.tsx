import { useState } from 'react';
import type { GameState, TrainingStartOptions } from '@/types/game';
import { ArrowLeft, Brain, Users, Target, Activity, TrendingUp, Layers } from 'lucide-react';

interface DashboardProps {
  gameState: GameState;
  onBack: () => void;
  onStartTraining: (options: TrainingStartOptions) => void;
  onEnterValley: () => void;
}

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white/5 rounded-lg p-4 border border-white/10 mb-4">
      <h3 className="text-sm font-bold text-white mb-3">{title}</h3>
      {children}
    </div>
  );
}

function StatItem({ label, value, color = 'text-gray-400' }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-white/5 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-xs font-mono ${color}`}>{value}</span>
    </div>
  );
}

export default function Dashboard({ gameState, onBack, onStartTraining, onEnterValley }: DashboardProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'agents' | 'environment' | 'training'>('overview');

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: Layers },
    { id: 'agents' as const, label: 'Agents', icon: Users },
    { id: 'environment' as const, label: 'Environment', icon: Target },
    { id: 'training' as const, label: 'Training', icon: Activity },
  ];

  const totalResources = gameState.agents.reduce((sum, a) =>
    sum + Object.values(a.resources).reduce((s, v) => s + v, 0), 0);

  const avgLevel = gameState.agents.length > 0
    ? (gameState.agents.reduce((s, a) => s + a.level, 0) / gameState.agents.length).toFixed(1)
    : '0';

  const avgEnergy = gameState.agents.length > 0
    ? (gameState.agents.reduce((s, a) => s + a.energy, 0) / gameState.agents.length).toFixed(0)
    : '0';

  const stateCounts = gameState.agents.reduce((acc, a) => {
    acc[a.state] = (acc[a.state] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="screen-scroll bg-black text-white">
      {/* Header */}
      <div className="border-b border-white/10 bg-white/5">
        <div className="max-w-6xl mx-auto px-6 py-4 screen-scroll-content">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button onClick={onBack} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div>
                <h1 className="text-xl font-bold flex items-center gap-2">
                  <Brain className="w-5 h-5 text-amber-400" />
                  Project Dashboard
                </h1>
                <p className="text-xs text-gray-500">Agent Valley - Multi-Agent RL Environment</p>
              </div>
            </div>
            <button
              onClick={onEnterValley}
              className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-black rounded-lg font-medium text-sm transition-all hover:scale-105"
            >
              Enter Valley
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-4 screen-scroll-content">
        {/* Tabs */}
        <div className="flex gap-1 mb-6 bg-white/5 rounded-lg p-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all flex-1 justify-center ${
                activeTab === tab.id ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === 'overview' && (
          <div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                <div className="text-amber-400 text-xs mb-1">Total Agents</div>
                <div className="text-2xl font-bold text-white">{gameState.agents.length}</div>
              </div>
              <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                <div className="text-green-400 text-xs mb-1">Total Resources</div>
                <div className="text-2xl font-bold text-white">{totalResources.toFixed(0)}</div>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <div className="text-blue-400 text-xs mb-1">Avg Level</div>
                <div className="text-2xl font-bold text-white">{avgLevel}</div>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
                <div className="text-purple-400 text-xs mb-1">Avg Energy</div>
                <div className="text-2xl font-bold text-white">{avgEnergy}%</div>
              </div>
            </div>

            <SectionCard title="Project Overview">
              <p className="text-sm text-gray-400 mb-3">
                Agent Valley is a multi-agent reinforcement learning environment where backend role agents
                (farmer, miner, builder, warrior) coordinate shared resources, while the canvas shows a
                clearly labeled visual simulation using OODA-style behavior.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Problem Statement</h4>
                  <p className="text-xs text-gray-500">
                    Build an environment where multiple LLM-based agents must learn to cooperate and compete
                    in a shared resource economy, with sparse rewards and long-horizon planning.
                  </p>
                </div>
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Environment</h4>
                  <p className="text-xs text-gray-500">
                    A dataset-driven valley with farmland, mine, forest, village, wilderness, plains,
                    river, hills, and coast regions, plus market shocks and environmental events.
                  </p>
                </div>
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Agent Capabilities</h4>
                  <p className="text-xs text-gray-500">
                    Each agent uses the OODA loop (Observe-Orient-Decide-Act) to make independent decisions.
                    They can gather, trade, build, explore, rest, cooperate, compete, and defend.
                  </p>
                </div>
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Reward Model</h4>
                  <p className="text-xs text-gray-500">
                    Backend training now runs against the Python OpenEnv environment and reports
                    task score, reward, final status, and optimizer metrics from JSONL logs.
                  </p>
                </div>
              </div>
            </SectionCard>

            <SectionCard title="Hackathon Compliance">
              <div className="space-y-2">
                <StatItem label="Theme" value="Multi-Agent Interactions + Long-Horizon Planning" color="text-amber-400" />
                <StatItem label="Problem Statement" value="Defined" color="text-green-400" />
                <StatItem label="Environment" value="5 Regions, Dynamic Events" color="text-green-400" />
                <StatItem label="Agent Capabilities" value="OODA Loop, 8 Actions" color="text-green-400" />
                <StatItem label="Tasks" value="Gather, Trade, Build, Explore, Cooperate, Compete" color="text-green-400" />
                <StatItem label="Reward Model" value="Python env reward components" color="text-green-400" />
                <StatItem label="Self-Improvement" value="Q-learning, neural policy, GRPO" color="text-green-400" />
              </div>
            </SectionCard>
          </div>
        )}

        {activeTab === 'agents' && (
          <div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {gameState.agents.map(agent => (
                <div key={agent.id} className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-colors">
                  <div className="flex items-center gap-3 mb-3">
                    <img src={agent.avatar} alt={agent.name} className="w-10 h-10 object-contain" />
                    <div>
                      <div className="text-sm font-bold text-white">{agent.name}</div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ backgroundColor: `${agent.color}30`, color: agent.color }}>
                          {agent.role}
                        </span>
                        <span className="text-[10px] text-gray-500">Lv.{agent.level}</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mb-2">
                    <div className="bg-black/30 rounded p-2 text-center">
                      <div className="text-[10px] text-gray-500">Health</div>
                      <div className="text-sm font-bold text-red-400">{Math.round(agent.health)}</div>
                    </div>
                    <div className="bg-black/30 rounded p-2 text-center">
                      <div className="text-[10px] text-gray-500">Energy</div>
                      <div className="text-sm font-bold text-yellow-400">{Math.round(agent.energy)}</div>
                    </div>
                    <div className="bg-black/30 rounded p-2 text-center">
                      <div className="text-[10px] text-gray-500">XP</div>
                      <div className="text-sm font-bold text-purple-400">{Math.round(agent.experience)}</div>
                    </div>
                  </div>

                  <div className="text-[10px] text-gray-500">
                    State: <span className="text-white capitalize">{agent.state}</span> |
                    Visual Actions: {agent.memory.successfulActions.length + agent.memory.failedActions.length} |
                    Visual Win Rate: {agent.memory.successfulActions.length + agent.memory.failedActions.length > 0
                      ? (agent.memory.successfulActions.length / (agent.memory.successfulActions.length + agent.memory.failedActions.length) * 100).toFixed(0)
                      : 0}%
                  </div>
                </div>
              ))}
            </div>

            {/* Agent state distribution */}
            <SectionCard title="Agent State Distribution">
              <div className="space-y-2">
                {Object.entries(stateCounts).map(([state, count]) => (
                  <div key={state} className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 w-20 capitalize">{state}</span>
                    <div className="flex-1 h-3 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-amber-500 transition-all"
                        style={{ width: `${(count / gameState.agents.length) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-8 text-right">{count}</span>
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        )}

        {activeTab === 'environment' && (
          <div>
            <SectionCard title="Market State">
              <div className="grid grid-cols-5 gap-2">
                {Object.entries(gameState.market.prices).map(([resource, price]) => (
                  <div key={resource} className="bg-black/30 rounded p-2 text-center">
                    <div className="text-lg">
                      {resource === 'food' ? '🌾' : resource === 'wood' ? '🪵' : resource === 'stone' ? '🪨' : resource === 'gold' ? '🪙' : '⛏️'}
                    </div>
                    <div className="text-[10px] text-gray-500 uppercase">{resource}</div>
                    <div className="text-sm font-bold text-amber-400">{price}</div>
                  </div>
                ))}
              </div>
            </SectionCard>

            <SectionCard title="Recent Events">
              {gameState.events.length === 0 ? (
                <p className="text-xs text-gray-600">No events yet. Start the simulation to see events.</p>
              ) : (
                <div className="space-y-1 max-h-64 overflow-y-auto">
                  {[...gameState.events].reverse().map((event, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-white/5">
                      <span className="text-gray-600">[{event.tick}]</span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase ${
                        event.type === 'disaster' || event.type === 'invasion' ? 'bg-red-500/20 text-red-400' :
                        event.type === 'resource_spawn' ? 'bg-green-500/20 text-green-400' :
                        event.type === 'quest' ? 'bg-purple-500/20 text-purple-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {event.type}
                      </span>
                      <span className="text-gray-400">{event.description}</span>
                    </div>
                  ))}
                </div>
              )}
            </SectionCard>

            <SectionCard title="Valley Regions">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  { name: 'Green Meadows', type: 'farmland', agents: 2, danger: 'Low', color: '#4CAF50' },
                  { name: 'Crystal Mines', type: 'mine', agents: 2, danger: 'Medium', color: '#FF9800' },
                  { name: 'Whispering Woods', type: 'forest', agents: 2, danger: 'Low', color: '#8BC34A' },
                  { name: 'Trader\'s Post', type: 'village', agents: 2, danger: 'Low', color: '#2196F3' },
                  { name: 'Wild Frontier', type: 'wilderness', agents: 0, danger: 'High', color: '#F44336' },
                ].map(region => (
                  <div key={region.name} className="bg-white/5 rounded-lg p-3 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: `${region.color}20` }}>
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: region.color }} />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-white">{region.name}</div>
                      <div className="text-[10px] text-gray-500">
                        Type: {region.type} | Danger: {region.danger}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        )}

        {activeTab === 'training' && (
          <div>
            <SectionCard title="Training Configuration">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Algorithm</h4>
                  <p className="text-xs text-gray-500">
                    The Python backend supports tabular Q-learning, a CPU neural policy, and a
                    GRPO-style clipped policy objective over composite action indices.
                  </p>
                </div>
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Training Loop</h4>
                  <p className="text-xs text-gray-500">
                    1. Reset AgentValleyEnv with a seed<br />
                    2. Select composite actions from the backend policy<br />
                    3. Step the real environment reward function<br />
                    4. Write metrics to backend JSONL artifacts<br />
                    5. Poll the FastAPI metrics endpoint
                  </p>
                </div>
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Reward Function</h4>
                  <p className="text-xs text-gray-500">
                    Progress, objective match, focus match, cooperation, risk posture, safety, loop,
                    and format components are computed server-side by the Python environment.
                  </p>
                </div>
                <div className="bg-white/5 rounded-lg p-3">
                  <h4 className="text-xs text-amber-400 mb-2">Self-Improvement</h4>
                  <p className="text-xs text-gray-500">
                    The animated agents remain a visual simulation. Backend learning curves come from
                    persisted training artifacts, not local animation memory.
                  </p>
                </div>
              </div>
            </SectionCard>

            <div className="flex gap-3 mb-6">
              <button
                onClick={() => onStartTraining({ mode: 'q_learning', episodes: 10, difficulty: 'mixed', seed: 42 })}
                className="flex-1 px-4 py-3 bg-green-600 hover:bg-green-500 text-white rounded-lg font-medium text-sm transition-all hover:scale-105 flex items-center justify-center gap-2"
              >
                <TrendingUp className="w-4 h-4" />
                Q-learning (10 episodes)
              </button>
              <button
                onClick={() => onStartTraining({ mode: 'grpo', episodes: 10, difficulty: 'mixed', seed: 42 })}
                className="flex-1 px-4 py-3 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-medium text-sm transition-all hover:scale-105 flex items-center justify-center gap-2"
              >
                <Activity className="w-4 h-4" />
                GRPO (10 episodes)
              </button>
              <button
                onClick={() => onStartTraining({ mode: 'multi_agent_grpo', episodes: 5, difficulty: 'mixed', seed: 42 })}
                className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium text-sm transition-all hover:scale-105 flex items-center justify-center gap-2"
              >
                <Users className="w-4 h-4" />
                Multi-agent GRPO
              </button>
            </div>

            <SectionCard title="Current Training Status">
              <div className="space-y-2">
                <StatItem label="Status" value={gameState.training.isTraining ? 'Active' : 'Idle'} color={gameState.training.isTraining ? 'text-green-400' : 'text-gray-400'} />
                <StatItem label="Episode" value={`${gameState.training.episode} / ${gameState.training.maxEpisodes}`} />
                <StatItem label="Best Reward" value={gameState.training.bestReward > -99999 ? gameState.training.bestReward.toFixed(2) : 'N/A'} color="text-amber-400" />
                <StatItem label="Average Reward" value={gameState.training.avgReward.toFixed(2)} />
                <StatItem label="Policy Loss" value={gameState.training.loss.toFixed(4)} />
              </div>
            </SectionCard>
          </div>
        )}
      </div>
    </div>
  );
}
