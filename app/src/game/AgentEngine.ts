// Agent Valley - Core Agent Engine with OODA Loop
import type { Agent, ActionType, ResourceType, Position, AgentRole, AgentState } from '@/types/game';

const ROLE_COLORS: Record<AgentRole, string> = {
  farmer: '#4CAF50',
  miner: '#FF9800',
  builder: '#8D6E63',
  warrior: '#F44336',
};

const ROLE_AVATARS: Record<AgentRole, string> = {
  farmer: '/assets/farmer.png',
  miner: '/assets/miner.png',
  builder: '/assets/builder.png',
  warrior: '/assets/warrior.png',
};

const ROLE_SKILLS: Record<AgentRole, Record<string, number>> = {
  farmer: { farming: 0.8, trading: 0.5, gathering: 0.7, building: 0.3, combat: 0.2 },
  miner: { mining: 0.9, gathering: 0.8, trading: 0.4, building: 0.5, combat: 0.4 },
  builder: { building: 0.9, crafting: 0.8, trading: 0.5, gathering: 0.4, combat: 0.3 },
  warrior: { combat: 0.9, scouting: 0.8, gathering: 0.5, trading: 0.2, building: 0.3 },
};

const NAMES = [
  'Eldrin', 'Thorne', 'Willow', 'Gareth', 'Mira', 'Orion',
  'Sylas', 'Freya', 'Cedric', 'Luna', 'Rex', 'Nova',
  'Kai', 'Iris', 'Zane', 'Ruby', 'Finn', 'Ember',
  'Ash', 'Ivy', 'Cole', 'Sky', 'Blaze', 'River',
];

function randomPos(): Position {
  return { x: 50 + Math.random() * 700, y: 100 + Math.random() * 400 };
}

function dist(a: Position, b: Position): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export class AgentEngine {
  static createAgent(role: AgentRole, index: number): Agent {
    const name = NAMES[index % NAMES.length];
    return {
      id: generateId(),
      name: `${name} the ${role.charAt(0).toUpperCase() + role.slice(1)}`,
      role,
      state: 'idle',
      position: randomPos(),
      resources: { food: 10, wood: 5, stone: 5, gold: 2, ore: 0 },
      health: 100,
      energy: 100,
      experience: 0,
      level: 1,
      avatar: ROLE_AVATARS[role],
      color: ROLE_COLORS[role],
      skills: { ...ROLE_SKILLS[role] },
      decisionPolicy: {
        explorationRate: 0.3,
        cooperationBias: Math.random() * 0.5 + 0.2,
        riskTolerance: Math.random() * 0.4 + 0.1,
        learningRate: 0.1,
        qValues: {},
      },
      ooda: {
        observe: { nearbyAgents: [], nearbyResources: [], threats: [], opportunities: [] },
        orient: { currentNeed: 'food', threatLevel: 0, opportunityScore: 0, personalityBias: Math.random() },
        decide: { chosenAction: 'gather', target: null, confidence: 0.5 },
        act: { executing: false, progress: 0, result: 'pending' },
      },
      memory: {
        observations: [],
        successfulActions: [],
        failedActions: [],
        cooperations: [],
        competitions: [],
      },
    };
  }

  static runOODA(agent: Agent, allAgents: Agent[], tick: number): { agent: Agent; action: ActionType } {
    // === O - OBSERVE ===
    const nearbyAgents = allAgents.filter(a => a.id !== agent.id && dist(a.position, agent.position) < 120);
    const nearbyResources: any[] = []; // Will be filled by environment
    const threats = nearbyAgents.filter(a => dist(a.position, agent.position) < 80 && a.health > agent.health);
    const opportunities: string[] = [];

    if (nearbyAgents.length > 0) opportunities.push('trade');
    if (threats.length > 0) opportunities.push('defend');
    if (agent.energy < 30) opportunities.push('rest');
    if (agent.resources.food < 5) opportunities.push('gather_food');

    agent.ooda.observe = { nearbyAgents, nearbyResources, threats, opportunities };

    // === O - ORIENT ===
    let currentNeed: ResourceType | 'safety' | 'wealth' | 'cooperation' = 'food';
    if (agent.health < 40) currentNeed = 'safety';
    else if (agent.energy < 20) currentNeed = 'safety';
    else if (agent.resources.food < 10) currentNeed = 'food';
    else if (agent.resources.gold < 5) currentNeed = 'wealth';
    else if (nearbyAgents.length > 1 && agent.decisionPolicy.cooperationBias > 0.5) currentNeed = 'cooperation';

    const threatLevel = threats.length * 0.3 + (agent.health < 50 ? 0.3 : 0);
    const opportunityScore = opportunities.length * 0.2 + agent.energy / 100;

    agent.ooda.orient = {
      currentNeed,
      threatLevel,
      opportunityScore,
      personalityBias: agent.decisionPolicy.cooperationBias,
    };

    // === D - DECIDE ===
    const possibleActions: ActionType[] = ['gather', 'trade', 'build', 'explore', 'rest', 'cooperate', 'compete', 'defend'];
    const weights = possibleActions.map(action => this.calculateActionWeight(agent, action, nearbyAgents));
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    const probs = weights.map(w => w / totalWeight);

    // Epsilon-greedy exploration
    let chosenAction: ActionType;
    if (Math.random() < agent.decisionPolicy.explorationRate) {
      chosenAction = possibleActions[Math.floor(Math.random() * possibleActions.length)];
    } else {
      const maxProb = Math.max(...probs);
      const idx = probs.indexOf(maxProb);
      chosenAction = possibleActions[idx];
    }

    const confidence = probs[possibleActions.indexOf(chosenAction)];
    agent.ooda.decide = { chosenAction, target: nearbyAgents[0]?.id || null, confidence };

    // === A - ACT ===
    agent.ooda.act = { executing: true, progress: 0, result: 'pending' };
    agent.state = this.actionToState(chosenAction);

    // Record observation
    agent.memory.observations.push({
      tick,
      what: `Decided to ${chosenAction}`,
      where: agent.position,
      importance: confidence,
    });
    if (agent.memory.observations.length > 50) agent.memory.observations.shift();

    return { agent, action: chosenAction };
  }

  static executeAction(agent: Agent, action: ActionType): Agent {
    const reward = this.calculateReward(agent, action);
    agent.energy = Math.max(0, agent.energy - 5);

    switch (action) {
      case 'gather':
        agent.resources.food += Math.floor(Math.random() * 5) + 1;
        agent.resources.wood += Math.floor(Math.random() * 3);
        agent.resources.stone += Math.floor(Math.random() * 2);
        agent.energy -= 5;
        break;
      case 'trade':
        if (agent.resources.food > 5) {
          agent.resources.food -= 3;
          agent.resources.gold += 2;
        }
        agent.energy -= 3;
        break;
      case 'build':
        if (agent.resources.wood >= 3 && agent.resources.stone >= 2) {
          agent.resources.wood -= 3;
          agent.resources.stone -= 2;
          agent.experience += 15;
        }
        agent.energy -= 10;
        break;
      case 'explore':
        agent.position.x += (Math.random() - 0.5) * 60;
        agent.position.y += (Math.random() - 0.5) * 60;
        agent.position.x = Math.max(20, Math.min(780, agent.position.x));
        agent.position.y = Math.max(60, Math.min(500, agent.position.y));
        agent.energy -= 8;
        break;
      case 'rest':
        agent.energy = Math.min(100, agent.energy + 30);
        agent.health = Math.min(100, agent.health + 10);
        break;
      case 'cooperate':
        agent.resources.food += 2;
        agent.experience += 10;
        agent.energy -= 5;
        break;
      case 'compete':
        if (Math.random() > 0.5) {
          agent.resources.gold += 3;
          agent.experience += 20;
        } else {
          agent.health -= 15;
          agent.energy -= 15;
        }
        break;
      case 'defend':
        agent.energy -= 8;
        agent.health = Math.min(100, agent.health + 5);
        break;
    }

    agent.experience += reward;
    if (agent.experience >= agent.level * 100) {
      agent.level += 1;
      agent.experience = 0;
    }

    agent.ooda.act = { executing: false, progress: 1, result: reward > 0 ? 'success' : 'failure' };

    if (reward > 0) {
      agent.memory.successfulActions.push({ action, reward, tick: Date.now() });
    } else {
      agent.memory.failedActions.push({ action, reward, tick: Date.now() });
    }

    // Update Q-values
    const actionKey = action;
    if (!agent.decisionPolicy.qValues[actionKey]) {
      agent.decisionPolicy.qValues[actionKey] = [];
    }
    agent.decisionPolicy.qValues[actionKey].push(reward);
    if (agent.decisionPolicy.qValues[actionKey].length > 20) {
      agent.decisionPolicy.qValues[actionKey].shift();
    }

    // Decay exploration
    agent.decisionPolicy.explorationRate *= 0.9995;

    return agent;
  }

  private static calculateActionWeight(agent: Agent, action: ActionType, nearby: Agent[]): number {
    const weights: Record<ActionType, number> = {
      gather: agent.resources.food < 20 ? 2.0 : 0.5,
      trade: nearby.length > 0 && agent.resources.food > 5 ? 1.5 : 0.2,
      build: agent.resources.wood >= 5 && agent.resources.stone >= 3 ? 1.2 : 0.3,
      explore: agent.energy > 60 ? 1.0 : 0.2,
      rest: agent.energy < 30 ? 3.0 : agent.energy < 60 ? 1.0 : 0.1,
      cooperate: nearby.length > 0 && agent.decisionPolicy.cooperationBias > 0.4 ? 1.8 : 0.3,
      compete: nearby.length > 0 && agent.decisionPolicy.cooperationBias < 0.3 ? 1.2 : 0.2,
      defend: agent.ooda.orient.threatLevel > 0.5 ? 2.5 : 0.1,
    };
    return weights[action] || 0.5;
  }

  private static calculateReward(_agent: Agent, action: ActionType): number {
    const baseRewards: Record<ActionType, number> = {
      gather: 5, trade: 8, build: 12, explore: 6,
      rest: 3, cooperate: 10, compete: 15, defend: 4,
    };
    return baseRewards[action] || 5;
  }

  private static actionToState(action: ActionType): AgentState {
    const map: Record<ActionType, AgentState> = {
      gather: 'working', trade: 'trading', build: 'working',
      explore: 'moving', rest: 'resting', cooperate: 'cooperating',
      compete: 'fighting', defend: 'fighting',
    };
    return map[action] || 'idle';
  }

  static moveAgent(agent: Agent, target: Position, speed: number = 2): Agent {
    const dx = target.x - agent.position.x;
    const dy = target.y - agent.position.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > speed) {
      agent.position.x += (dx / d) * speed;
      agent.position.y += (dy / d) * speed;
      agent.state = 'moving';
    }
    return agent;
  }
}
