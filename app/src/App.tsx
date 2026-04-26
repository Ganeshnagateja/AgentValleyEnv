import { useGameLoop } from '@/hooks/useGameLoop';
import TitleScreen from '@/sections/TitleScreen';
import GameCanvas from '@/sections/GameCanvas';
import AgentPanel from '@/sections/AgentPanel';
import TrainingDashboard from '@/sections/TrainingDashboard';
import Dashboard from '@/sections/Dashboard';
import { BarChart3, LayoutDashboard, BookOpen } from 'lucide-react';

function App() {
  const {
    gameState,
    pauseGame,
    resumeGame,
    setSpeed,
    selectAgent,
    setView,
    startTraining,
    stopTraining,
    resetSimulation,
    enterTheValley,
  } = useGameLoop();

  // Title screen
  if (gameState.view === 'title') {
    return (
      <TitleScreen
        onEnterValley={enterTheValley}
        onViewTraining={() => setView('training')}
        onViewDashboard={() => setView('dashboard')}
      />
    );
  }

  // Training dashboard
  if (gameState.view === 'training') {
    return (
      <TrainingDashboard
        training={gameState.training}
        agents={gameState.agents}
        onStartTraining={startTraining}
        onStopTraining={stopTraining}
        onReset={resetSimulation}
        onBack={() => setView('title')}
      />
    );
  }

  // Project dashboard
  if (gameState.view === 'dashboard') {
    return (
      <Dashboard
        gameState={gameState}
        onBack={() => setView('title')}
        onStartTraining={startTraining}
        onEnterValley={enterTheValley}
      />
    );
  }

  // Main game view
  const selectedAgent = gameState.agents.find(a => a.id === gameState.selectedAgent) || null;

  return (
    <div className="h-screen w-screen bg-black flex overflow-hidden">
      {/* Main game area */}
      <div className="flex-1 relative">
        <GameCanvas
          agents={gameState.agents}
          resources={gameState.resources}
          isRunning={gameState.isRunning}
          speed={gameState.speed}
          selectedAgent={gameState.selectedAgent}
          onSelectAgent={selectAgent}
          onPause={pauseGame}
          onResume={resumeGame}
          onSpeedChange={setSpeed}
        />

        {/* Bottom navigation bar */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-black/80 backdrop-blur-md rounded-xl px-4 py-2 border border-white/10">
          <button
            onClick={() => setView('title')}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors text-gray-400 hover:text-white"
            title="Main Menu"
          >
            <BookOpen className="w-4 h-4" />
          </button>
          <div className="w-px h-5 bg-white/20" />
          <button
            onClick={() => setView('training')}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors text-gray-400 hover:text-white"
            title="Training Lab"
          >
            <BarChart3 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setView('dashboard')}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors text-gray-400 hover:text-white"
            title="Dashboard"
          >
            <LayoutDashboard className="w-4 h-4" />
          </button>
          <div className="w-px h-5 bg-white/20" />
          <span className="text-[10px] text-gray-500">
            Tick: {gameState.tick}
          </span>
        </div>
      </div>

      {/* Right side agent panel */}
      <AgentPanel
        agent={selectedAgent}
        onClose={() => selectAgent(null)}
      />
    </div>
  );
}

export default App;
