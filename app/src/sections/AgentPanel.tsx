import type { Agent } from '@/types/game';
import { X, Heart, Zap, Star, TrendingUp, Brain, Eye, Compass, Swords, Activity } from 'lucide-react';

interface AgentPanelProps {
  agent: Agent | null;
  onClose: () => void;
}

function StatBar({ label, value, max, color, icon: Icon }: {
  label: string; value: number; max: number; color: string; icon: any;
}) {
  const pct = (value / max) * 100;
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-1.5">
          <Icon className="w-3 h-3 text-gray-500" />
          <span className="text-xs text-gray-400">{label}</span>
        </div>
        <span className="text-xs font-mono text-white">{Math.round(value)}/{max}</span>
      </div>
      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function ResourceGrid({ resources }: { resources: Record<string, number> }) {
  const icons: Record<string, string> = {
    food: '🌾', wood: '🪵', stone: '🪨', gold: '🪙', ore: '⛏️',
  };

  return (
    <div className="grid grid-cols-5 gap-1">
      {Object.entries(resources).map(([type, amount]) => (
        <div key={type} className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className="text-lg">{icons[type]}</div>
          <div className="text-[10px] text-gray-500 uppercase">{type}</div>
          <div className="text-sm font-bold text-white">{Math.round(amount)}</div>
        </div>
      ))}
    </div>
  );
}

function OODACircle({ stage, active, label }: { stage: string; active: boolean; label: string }) {
  const colors: Record<string, string> = {
    observe: '#2196F3',
    orient: '#FF9800',
    decide: '#4CAF50',
    act: '#F44336',
  };

  return (
    <div className={`flex flex-col items-center transition-all duration-300 ${active ? 'scale-110' : 'opacity-50'}`}>
      <div
        className="w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all"
        style={{
          borderColor: colors[stage] || '#666',
          backgroundColor: active ? `${colors[stage]}30` : 'transparent',
          boxShadow: active ? `0 0 12px ${colors[stage]}60` : 'none',
        }}
      >
        <span className="text-[10px] font-bold text-white uppercase">{stage[0]}</span>
      </div>
      <span className="text-[9px] text-gray-400 mt-1 uppercase">{label}</span>
    </div>
  );
}

function VisualActionScoreChart({ qValues }: { qValues: Record<string, number[]> }) {
  const actions = Object.keys(qValues);
  if (actions.length === 0) return null;

  const maxVal = Math.max(...actions.flatMap(a => qValues[a]), 1);

  return (
    <div className="bg-white/5 rounded-lg p-3">
      <h4 className="text-xs text-gray-400 mb-2 flex items-center gap-1">
        <TrendingUp className="w-3 h-3" /> Visual Action Scores
      </h4>
      <div className="space-y-1.5">
        {actions.map(action => {
          const vals = qValues[action];
          const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
          const pct = (avg / maxVal) * 100;
          return (
            <div key={action} className="flex items-center gap-2">
              <span className="text-[9px] text-gray-500 w-16 uppercase">{action}</span>
              <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-500 rounded-full transition-all"
                  style={{ width: `${Math.min(100, Math.max(5, pct))}%` }}
                />
              </div>
              <span className="text-[9px] text-gray-400 w-8 text-right">{avg.toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function AgentPanel({ agent, onClose }: AgentPanelProps) {
  if (!agent) {
    return (
      <div className="w-72 bg-black/90 border-l border-white/10 flex flex-col items-center justify-center p-6">
        <Brain className="w-12 h-12 text-gray-700 mb-3" />
        <p className="text-gray-500 text-sm text-center">
          Select an agent on the map to view their OODA loop, stats, and decision history
        </p>
        <div className="mt-4 flex items-center gap-2 text-gray-600 text-xs">
          <Eye className="w-3 h-3" />
          <span>Click any character</span>
        </div>
      </div>
    );
  }

  const stateColors: Record<string, string> = {
    idle: '#8BC34A', working: '#4CAF50', trading: '#2196F3',
    fighting: '#F44336', resting: '#9E9E9E', moving: '#FF9800', cooperating: '#E040FB',
  };

  const recentObservations = agent.memory.observations.slice(-5).reverse();
  const successCount = agent.memory.successfulActions.length;
  const failCount = agent.memory.failedActions.length;
  const winRate = successCount + failCount > 0 ? (successCount / (successCount + failCount) * 100).toFixed(0) : '0';

  return (
    <div className="w-80 bg-black/95 border-l border-white/10 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <img src={agent.avatar} alt={agent.name} className="w-12 h-12 object-contain" />
            <div>
              <h3 className="text-white font-bold text-sm">{agent.name}</h3>
              <div className="flex items-center gap-2 mt-0.5">
                <span
                  className="text-[10px] px-2 py-0.5 rounded-full uppercase font-bold"
                  style={{
                    backgroundColor: `${agent.color}30`,
                    color: agent.color,
                    border: `1px solid ${agent.color}60`,
                  }}
                >
                  {agent.role}
                </span>
                <span className="text-[10px] text-gray-500">Lv.{agent.level}</span>
              </div>
            </div>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
            <X className="w-4 h-4 text-gray-500" />
          </button>
        </div>

        {/* State badge */}
        <div className="mt-3 flex items-center gap-2">
          <Activity className="w-3 h-3 text-gray-500" />
          <span
            className="text-[10px] px-2 py-0.5 rounded uppercase font-bold"
            style={{ backgroundColor: `${stateColors[agent.state] || '#666'}30`, color: stateColors[agent.state] || '#aaa' }}
          >
            {agent.state}
          </span>
          <span className="text-gray-600 text-[10px]">Win Rate: {winRate}%</span>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* OODA Loop Visualization */}
        <div>
          <h4 className="text-xs text-gray-400 mb-3 flex items-center gap-1">
            <Brain className="w-3 h-3" /> OODA Loop
          </h4>
          <div className="flex items-center justify-between">
            <OODACircle stage="observe" active={agent.state !== 'idle'} label="Observe" />
            <div className="w-4 h-px bg-white/10" />
            <OODACircle stage="orient" active={agent.ooda.orient.threatLevel > 0} label="Orient" />
            <div className="w-4 h-px bg-white/10" />
            <OODACircle stage="decide" active={agent.ooda.decide.confidence > 0.5} label="Decide" />
            <div className="w-4 h-px bg-white/10" />
            <OODACircle stage="act" active={agent.ooda.act.executing} label="Act" />
          </div>

          {/* Current decision */}
          {agent.ooda.decide.chosenAction && (
            <div className="mt-2 bg-white/5 rounded-lg p-2 text-center">
              <span className="text-[10px] text-gray-500 uppercase">Current Action</span>
              <div className="text-sm font-bold text-amber-400">{agent.ooda.decide.chosenAction}</div>
              <div className="text-[10px] text-gray-500">
                Confidence: {(agent.ooda.decide.confidence * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>

        {/* Stats */}
        <div>
          <StatBar label="Health" value={agent.health} max={100} color="#F44336" icon={Heart} />
          <StatBar label="Energy" value={agent.energy} max={100} color="#FFEB3B" icon={Zap} />
          <StatBar label="Experience" value={agent.experience} max={agent.level * 100} color="#9C27B0" icon={Star} />
        </div>

        {/* Resources */}
        <div>
          <h4 className="text-xs text-gray-400 mb-2">Inventory</h4>
          <ResourceGrid resources={agent.resources} />
        </div>

        {/* Skills */}
        <div className="bg-white/5 rounded-lg p-3">
          <h4 className="text-xs text-gray-400 mb-2 flex items-center gap-1">
            <Swords className="w-3 h-3" /> Skills
          </h4>
          <div className="space-y-1.5">
            {Object.entries(agent.skills).map(([skill, value]) => (
              <div key={skill} className="flex items-center gap-2">
                <span className="text-[9px] text-gray-500 w-14 capitalize">{skill}</span>
                <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${value * 100}%`,
                      backgroundColor: value > 0.7 ? '#4CAF50' : value > 0.4 ? '#FF9800' : '#F44336',
                    }}
                  />
                </div>
                <span className="text-[9px] text-gray-400 w-6 text-right">{(value * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Visual-only action scores from the canvas simulation. Backend Q-learning is shown in Training. */}
        <VisualActionScoreChart qValues={agent.decisionPolicy.qValues} />

        {/* Decision Policy */}
        <div className="bg-white/5 rounded-lg p-3">
          <h4 className="text-xs text-gray-400 mb-2">
            <Compass className="w-3 h-3 inline mr-1" /> Decision Policy
          </h4>
          <div className="space-y-1 text-[10px]">
            <div className="flex justify-between">
              <span className="text-gray-500">Exploration</span>
              <span className="text-amber-400">{(agent.decisionPolicy.explorationRate * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Cooperation Bias</span>
              <span className="text-blue-400">{(agent.decisionPolicy.cooperationBias * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Risk Tolerance</span>
              <span className="text-red-400">{(agent.decisionPolicy.riskTolerance * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Learning Rate</span>
              <span className="text-green-400">{(agent.decisionPolicy.learningRate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Recent Observations */}
        <div className="bg-white/5 rounded-lg p-3">
          <h4 className="text-xs text-gray-400 mb-2 flex items-center gap-1">
            <Eye className="w-3 h-3" /> Recent Observations
          </h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {recentObservations.length === 0 ? (
              <span className="text-gray-600 text-[10px]">No observations yet...</span>
            ) : (
              recentObservations.map((obs, i) => (
                <div key={i} className="text-[10px] text-gray-400 flex items-start gap-1">
                  <span className="text-gray-600">[{obs.tick}]</span>
                  <span>{obs.what}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
