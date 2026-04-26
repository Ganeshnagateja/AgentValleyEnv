import { useRef, useEffect, useState, useCallback } from 'react';
import type { Agent, Resource } from '@/types/game';
import { Pause, Play, Zap, Wind } from 'lucide-react';

interface GameCanvasProps {
  agents: Agent[];
  resources: Resource[];
  isRunning: boolean;
  speed: number;
  selectedAgent: string | null;
  onSelectAgent: (id: string | null) => void;
  onPause: () => void;
  onResume: () => void;
  onSpeedChange: (speed: number) => void;
}

function AgentSprite({ agent, isSelected, onClick }: { agent: Agent; isSelected: boolean; onClick: () => void }) {
  const [imgLoaded, setImgLoaded] = useState(false);
  const [bounceY, setBounceY] = useState(0);

  useEffect(() => {
    const img = new Image();
    img.src = agent.avatar;
    img.onload = () => setImgLoaded(true);
  }, [agent.avatar]);

  // Bobbing animation based on agent state
  useEffect(() => {
    let animId: number;
    let time = 0;
    const animate = () => {
      time += 0.05;
      const bobAmount = agent.state === 'moving' ? 3 : agent.state === 'working' ? 1.5 : 0.5;
      const bobSpeed = agent.state === 'moving' ? 8 : 3;
      setBounceY(Math.sin(time * bobSpeed) * bobAmount);
      animId = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animId);
  }, [agent.state]);

  return (
    <div
      className={`absolute transition-all cursor-pointer group ${isSelected ? 'z-30' : 'z-20'}`}
      style={{
        left: agent.position.x - 25,
        top: agent.position.y - 45 + bounceY,
        width: 50,
        height: 60,
      }}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
    >
      {/* Selection ring */}
      {isSelected && (
        <div className="absolute -inset-2 border-2 border-amber-400 rounded-full animate-pulse" 
          style={{ boxShadow: '0 0 15px rgba(251,191,36,0.6)' }} />
      )}

      {/* Agent image */}
      {imgLoaded ? (
        <img
          src={agent.avatar}
          alt={agent.name}
          className="w-full h-full object-contain drop-shadow-lg transition-transform group-hover:scale-110"
          style={{
            filter: agent.energy < 30 ? 'grayscale(0.5) brightness(0.7)' : 'none',
          }}
        />
      ) : (
        <div
          className="w-full h-full rounded-full animate-pulse"
          style={{ backgroundColor: agent.color }}
        />
      )}

      {/* Name tag */}
      <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
        <span
          className="text-[10px] font-bold px-1.5 py-0.5 rounded-full text-white"
          style={{ backgroundColor: agent.color, textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}
        >
          {agent.name}
        </span>
      </div>

      {/* State indicator */}
      <div
        className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full"
        style={{
          backgroundColor:
            agent.state === 'working' ? '#4CAF50' :
            agent.state === 'trading' ? '#2196F3' :
            agent.state === 'fighting' ? '#F44336' :
            agent.state === 'resting' ? '#9E9E9E' :
            agent.state === 'moving' ? '#FF9800' :
            agent.state === 'cooperating' ? '#E040FB' :
            '#8BC34A',
          boxShadow: `0 0 6px ${agent.color}`,
        }}
      />
    </div>
  );
}

function ResourceNode({ resource }: { resource: Resource }) {
  const colors: Record<string, string> = {
    food: '#4CAF50',
    wood: '#8D6E63',
    stone: '#78909C',
    gold: '#FFD700',
    ore: '#FF5722',
  };

  return (
    <div
      className="absolute z-10"
      style={{ left: resource.position.x - 6, top: resource.position.y - 6 }}
    >
      <div
        className="w-3 h-3 rounded-full animate-pulse"
        style={{
          backgroundColor: colors[resource.type] || '#888',
          boxShadow: `0 0 8px ${colors[resource.type] || '#888'}`,
        }}
      />
    </div>
  );
}

function RegionLabel({ name, x, y, type }: { name: string; x: number; y: number; type: string }) {
  const colors: Record<string, string> = {
    farmland: 'rgba(76,175,80,0.15)',
    mine: 'rgba(255,152,0,0.15)',
    forest: 'rgba(139,195,74,0.15)',
    village: 'rgba(33,150,243,0.15)',
    wilderness: 'rgba(244,67,54,0.15)',
  };

  return (
    <div
      className="absolute z-0 rounded-full"
      style={{
        left: x - 50,
        top: y - 50,
        width: 100,
        height: 100,
        backgroundColor: colors[type] || 'rgba(255,255,255,0.05)',
        border: `1px dashed ${colors[type]?.replace('0.15', '0.3') || 'rgba(255,255,255,0.1)'}`,
      }}
    >
      <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] text-white/40 whitespace-nowrap uppercase tracking-wider">
        {name}
      </span>
    </div>
  );
}

export default function GameCanvas({
  agents,
  resources,
  isRunning,
  speed,
  selectedAgent,
  onSelectAgent,
  onPause,
  onResume,
  onSpeedChange,
}: GameCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [showTrail, setShowTrail] = useState(true);

  const handleCanvasClick = useCallback(() => {
    onSelectAgent(null);
  }, [onSelectAgent]);

  return (
    <div className="relative w-full h-full">
      {/* Game Canvas */}
      <div
        ref={canvasRef}
        className="relative w-full h-full overflow-hidden cursor-crosshair"
        style={{
          backgroundImage: 'url(/assets/valley-bg.jpg)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
        onClick={handleCanvasClick}
      >
        {/* Dark overlay for better visibility */}
        <div className="absolute inset-0 bg-black/20" />

        {/* Region indicators */}
        <RegionLabel name="Green Meadows" x={100} y={150} type="farmland" />
        <RegionLabel name="Crystal Mines" x={600} y={120} type="mine" />
        <RegionLabel name="Whispering Woods" x={150} y={350} type="forest" />
        <RegionLabel name="Trader's Post" x={450} y={280} type="village" />
        <RegionLabel name="Wild Frontier" x={700} y={380} type="wilderness" />

        {/* Resources */}
        {resources.map((res, i) => (
          <ResourceNode key={`${res.type}-${i}`} resource={res} />
        ))}

        {/* Agents */}
        {agents.map(agent => (
          <AgentSprite
            key={agent.id}
            agent={agent}
            isSelected={selectedAgent === agent.id}
            onClick={() => onSelectAgent(agent.id)}
          />
        ))}

        {/* Grid overlay (subtle) */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-5">
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="white" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      {/* Controls HUD */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
        {/* Left: Status */}
        <div className="pointer-events-auto flex items-center gap-2 bg-black/60 backdrop-blur-md rounded-lg px-4 py-2 border border-white/10">
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-amber-400'}`} />
          <span className="text-white text-sm font-medium">
            {isRunning ? 'Simulation Running' : 'Paused'}
          </span>
          <span className="text-gray-500 text-xs ml-2">
            {agents.length} Agents Active
          </span>
        </div>

        {/* Right: Controls */}
        <div className="pointer-events-auto flex items-center gap-2 bg-black/60 backdrop-blur-md rounded-lg px-4 py-2 border border-white/10">
          <button
            onClick={isRunning ? onPause : onResume}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            {isRunning ? <Pause className="w-4 h-4 text-white" /> : <Play className="w-4 h-4 text-green-400" />}
          </button>

          <div className="w-px h-5 bg-white/20" />

          {[0.5, 1, 2, 4].map(s => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                speed === s ? 'bg-amber-500 text-black' : 'text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              {s}x
            </button>
          ))}

          <div className="w-px h-5 bg-white/20" />

          <button
            onClick={() => setShowTrail(!showTrail)}
            className={`p-2 rounded-lg transition-colors ${showTrail ? 'bg-white/20' : 'hover:bg-white/10'}`}
            title="Toggle trails"
          >
            <Wind className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Speed indicator */}
      {speed > 1 && (
        <div className="absolute top-16 left-4 bg-amber-500/20 border border-amber-500/40 rounded-lg px-3 py-1">
          <div className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-amber-400" />
            <span className="text-amber-400 text-xs font-bold">{speed}x Speed</span>
          </div>
        </div>
      )}
    </div>
  );
}
