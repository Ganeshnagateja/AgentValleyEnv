// Agent Valley - Multi-Agent Environment Engine
import type { Agent, Resource, EnvironmentEvent, ValleyRegion, MarketState, ResourceType, Position } from '@/types/game';

const REGION_DEFS = [
  { id: 'farmland', name: 'Green Meadows', type: 'farmland' as const, x: 100, y: 150, size: 80 },
  { id: 'mine', name: 'Crystal Mines', type: 'mine' as const, x: 600, y: 120, size: 70 },
  { id: 'forest', name: 'Whispering Woods', type: 'forest' as const, x: 150, y: 350, size: 90 },
  { id: 'village', name: 'Trader\'s Post', type: 'village' as const, x: 450, y: 280, size: 60 },
  { id: 'wilderness', name: 'Wild Frontier', type: 'wilderness' as const, x: 700, y: 380, size: 100 },
];

const RESOURCE_SPAWNS: Record<string, ResourceType[]> = {
  farmland: ['food'],
  mine: ['stone', 'ore', 'gold'],
  forest: ['wood', 'food'],
  village: ['gold'],
  wilderness: ['food', 'gold', 'ore'],
};

export class EnvironmentEngine {
  regions: ValleyRegion[];
  resources: Resource[];
  events: EnvironmentEvent[];
  market: MarketState;
  tick: number;

  constructor() {
    this.regions = REGION_DEFS.map(r => ({
      ...r,
      position: { x: r.x, y: r.y },
      resources: [],
      agents: [],
      danger: r.type === 'wilderness' ? 0.7 : r.type === 'mine' ? 0.4 : 0.1,
    }));
    this.resources = [];
    this.events = [];
    this.market = this.initMarket();
    this.tick = 0;
    this.spawnInitialResources();
  }

  private initMarket(): MarketState {
    return {
      prices: { food: 10, wood: 15, stone: 20, gold: 50, ore: 35 },
      history: [],
      volatility: 0.1,
    };
  }

  private spawnInitialResources() {
    for (const region of this.regions) {
      const types = RESOURCE_SPAWNS[region.type] || ['food'];
      for (const type of types) {
        for (let i = 0; i < 3; i++) {
          this.resources.push({
            type,
            amount: Math.floor(Math.random() * 20) + 5,
            position: {
              x: region.position.x + (Math.random() - 0.5) * region.size,
              y: region.position.y + (Math.random() - 0.5) * region.size,
            },
          });
        }
      }
    }
  }

  updateMarket() {
    const { prices, volatility } = this.market;
    for (const key of Object.keys(prices) as ResourceType[]) {
      const change = (Math.random() - 0.5) * 2 * volatility * prices[key];
      prices[key] = Math.max(1, Math.round(prices[key] + change));
    }
    this.market.history.push({ ...prices });
    if (this.market.history.length > 100) this.market.history.shift();
  }

  spawnResources() {
    if (Math.random() < 0.3) {
      const region = this.regions[Math.floor(Math.random() * this.regions.length)];
      const types = RESOURCE_SPAWNS[region.type] || ['food'];
      const type = types[Math.floor(Math.random() * types.length)];
      this.resources.push({
        type,
        amount: Math.floor(Math.random() * 15) + 5,
        position: {
          x: region.position.x + (Math.random() - 0.5) * region.size,
          y: region.position.y + (Math.random() - 0.5) * region.size,
        },
      });
    }
    if (this.resources.length > 80) {
      this.resources = this.resources.slice(-60);
    }
  }

  generateEvent(): EnvironmentEvent | null {
    if (Math.random() < 0.05) {
      const eventTypes: Array<'resource_spawn' | 'disaster' | 'market_fluctuation' | 'quest' | 'invasion'> = [
        'resource_spawn', 'disaster', 'market_fluctuation', 'quest', 'invasion'
      ];
      const type = eventTypes[Math.floor(Math.random() * eventTypes.length)];
      const descriptions: Record<string, string[]> = {
        resource_spawn: ['Rich veins discovered!', 'Bumper harvest season!', 'Mysterious cache found!'],
        disaster: ['Drought hits the valley!', 'Cave-in at the mines!', 'Plague of locusts!'],
        market_fluctuation: ['Trade caravan arrives!', 'Market crash!', 'Gold rush!'],
        quest: ['Ancient ruins discovered!', 'Lost treasure rumored!', 'Mystical portal opens!'],
        invasion: ['Bandit raid!', 'Wild beasts attack!', 'Rival faction spotted!'],
      };
      const desc = descriptions[type][Math.floor(Math.random() * descriptions[type].length)];
      const event: EnvironmentEvent = {
        type,
        description: desc,
        position: { x: Math.random() * 700 + 50, y: Math.random() * 400 + 80 },
        severity: Math.random(),
        tick: this.tick,
      };
      this.events.push(event);
      if (this.events.length > 20) this.events.shift();
      return event;
    }
    return null;
  }

  getRegionAt(pos: Position): ValleyRegion | null {
    return this.regions.find(r => {
      const d = Math.sqrt((pos.x - r.position.x) ** 2 + (pos.y - r.position.y) ** 2);
      return d < r.size;
    }) || null;
  }

  tickUpdate(agents: Agent[]) {
    this.tick++;
    this.spawnResources();
    this.updateMarket();
    const event = this.generateEvent();

    // Update region agents
    for (const region of this.regions) {
      region.agents = agents
        .filter(a => {
          const d = Math.sqrt((a.position.x - region.position.x) ** 2 + (a.position.y - region.position.y) ** 2);
          return d < region.size;
        })
        .map(a => a.id);
    }

    return { event, market: this.market };
  }

  getAgentContext(agent: Agent) {
    const region = this.getRegionAt(agent.position);
    const nearbyResources = this.resources.filter(r => {
      const d = Math.sqrt((r.position.x - agent.position.x) ** 2 + (r.position.y - agent.position.y) ** 2);
      return d < 80;
    });
    return { region, nearbyResources };
  }
}
