import { useEffect, useRef, useState } from 'react';
import { Play, BookOpen, BarChart3, Settings, ChevronRight, Sparkles, Users, Brain, Target } from 'lucide-react';

interface TitleScreenProps {
  onEnterValley: () => void;
  onViewTraining: () => void;
  onViewDashboard: () => void;
}

function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: Array<{
      x: number; y: number; size: number; speedY: number;
      speedX: number; opacity: number; hue: number;
    }> = [];

    for (let i = 0; i < 120; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 3 + 1,
        speedY: -Math.random() * 0.8 - 0.1,
        speedX: (Math.random() - 0.5) * 0.4,
        opacity: Math.random() * 0.6 + 0.2,
        hue: Math.random() * 60 + 180,
      });
    }

    let animId: number;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const p of particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, 70%, ${p.opacity})`;
        ctx.fill();

        // Glow effect
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 3);
        grad.addColorStop(0, `hsla(${p.hue}, 80%, 70%, ${p.opacity * 0.3})`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fill();

        p.y += p.speedY;
        p.x += p.speedX;

        if (p.y < -10) {
          p.y = canvas.height + 10;
          p.x = Math.random() * canvas.width;
        }
        if (p.x < -10) p.x = canvas.width + 10;
        if (p.x > canvas.width + 10) p.x = -10;
      }

      animId = requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />;
}

function FloatingCard({ icon: Icon, title, desc, delay }: { icon: any; title: string; desc: string; delay: number }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setShow(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  return (
    <div
      className={`bg-black/40 backdrop-blur-md border border-white/10 rounded-xl p-4 transition-all duration-700 hover:bg-white/10 hover:border-white/30 hover:scale-105 cursor-default ${
        show ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}
    >
      <Icon className="w-8 h-8 text-amber-400 mb-2" />
      <h3 className="text-white font-bold text-sm">{title}</h3>
      <p className="text-gray-400 text-xs mt-1">{desc}</p>
    </div>
  );
}

export default function TitleScreen({ onEnterValley, onViewTraining, onViewDashboard }: TitleScreenProps) {
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setLoaded(true), 200);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-black">
      {/* Background */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: 'url(/assets/title-bg.jpg)' }}
      >
        <div className="absolute inset-0 bg-black/40" />
      </div>

      {/* Particle overlay */}
      <ParticleCanvas />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4">
        {/* Title */}
        <div
          className={`text-center transition-all duration-1000 ${loaded ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-10'}`}
        >
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="w-6 h-6 text-amber-400 animate-pulse" />
            <span className="text-amber-400 text-sm font-medium tracking-[0.3em] uppercase">
              Multi-Agent RL Environment
            </span>
            <Sparkles className="w-6 h-6 text-amber-400 animate-pulse" />
          </div>

          <h1 className="text-6xl md:text-8xl font-black text-white mb-2 tracking-tight"
            style={{
              textShadow: '0 0 40px rgba(251,191,36,0.5), 0 0 80px rgba(251,191,36,0.2), 0 4px 20px rgba(0,0,0,0.8)',
            }}
          >
            AGENT VALLEY
          </h1>

          <p className="text-gray-300 text-lg md:text-xl max-w-2xl mx-auto mb-2"
            style={{ textShadow: '0 2px 10px rgba(0,0,0,0.8)' }}
          >
            Where autonomous agents learn, cooperate, and evolve through reinforcement learning
          </p>

          <div className="flex items-center justify-center gap-4 text-xs text-gray-500 mb-10">
            <span className="flex items-center gap-1"><Users className="w-3 h-3" /> 8 AI Agents</span>
            <span className="text-gray-600">|</span>
            <span className="flex items-center gap-1"><Brain className="w-3 h-3" /> OODA Loop</span>
            <span className="text-gray-600">|</span>
            <span className="flex items-center gap-1"><Target className="w-3 h-3" /> GRPO Training</span>
          </div>
        </div>

        {/* Main buttons */}
        <div
          className={`flex flex-col items-center gap-3 transition-all duration-1000 delay-300 ${loaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}
        >
          <button
            onClick={onEnterValley}
            className="group relative px-12 py-4 bg-amber-500 hover:bg-amber-400 text-black font-bold text-lg rounded-full transition-all duration-300 hover:scale-105 hover:shadow-[0_0_40px_rgba(251,191,36,0.5)] flex items-center gap-3"
          >
            <Play className="w-6 h-6 group-hover:scale-110 transition-transform" />
            Enter the Valley
            <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>

          <div className="flex gap-3 mt-4">
            <button
              onClick={onViewTraining}
              className="px-6 py-2.5 bg-white/10 hover:bg-white/20 border border-white/20 text-white rounded-lg transition-all hover:scale-105 flex items-center gap-2 text-sm backdrop-blur-sm"
            >
              <BarChart3 className="w-4 h-4" />
              Training Lab
            </button>
            <button
              onClick={onViewDashboard}
              className="px-6 py-2.5 bg-white/10 hover:bg-white/20 border border-white/20 text-white rounded-lg transition-all hover:scale-105 flex items-center gap-2 text-sm backdrop-blur-sm"
            >
              <BookOpen className="w-4 h-4" />
              Dashboard
            </button>
            <button className="px-6 py-2.5 bg-white/10 hover:bg-white/20 border border-white/20 text-white rounded-lg transition-all hover:scale-105 flex items-center gap-2 text-sm backdrop-blur-sm">
              <Settings className="w-4 h-4" />
              Settings
            </button>
          </div>
        </div>

        {/* Feature cards */}
        <div
          className={`grid grid-cols-2 md:grid-cols-4 gap-3 mt-16 max-w-3xl w-full px-4 transition-all duration-1000 delay-500 ${loaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}
        >
          <FloatingCard icon={Brain} title="OODA Loop" desc="Observe, Orient, Decide, Act cycle for every agent" delay={600} />
          <FloatingCard icon={Users} title="Multi-Agent" desc="8 agents with unique roles and personalities" delay={700} />
          <FloatingCard icon={Target} title="GRPO Training" desc="Group Relative Policy Optimization" delay={800} />
          <FloatingCard icon={BarChart3} title="Real-time Analytics" desc="Live training metrics and graphs" delay={900} />
        </div>

        {/* Footer */}
        <div className="absolute bottom-4 text-gray-600 text-xs">
          OpenEnv Hackathon 2026 | Multi-Agent RL Environment
        </div>
      </div>
    </div>
  );
}
