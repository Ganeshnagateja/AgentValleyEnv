# Agent Valley Writeup

## Problem Statement

Agent Valley is a multi-agent resource-management benchmark for Theme #1:
Multi-Agent Interactions. Four agents must keep a valley alive while handling
food shortages, market shocks, infrastructure pressure, and external threats.
The interesting behavior is not whether one agent can pick a good action in
isolation, but whether a team can divide labor, protect shared resources, and
avoid selfish pile-on behavior under partial observability.

## Environment Design

The project preserves the original OpenEnv-compatible
`env.environment.AgentValleyEnv` with `reset()`, `step(action)`, and `state()`.
That environment loads deterministic curriculum episodes from `datasets/` and
returns a machine-readable final status: `valley_stabilized`,
`valley_survived`, or `valley_collapsed`.

The Theme #1 layer is `env.multi_agent_env.MultiAgentValleyEnv`. It wraps the
single-agent environment and exposes four simultaneous agents:

- farmer: food producer in farmland
- miner: ore extractor in mine
- builder: infrastructure specialist in village
- warrior: defender in wilderness

Each agent receives a role-specific partial observation. Global risk and team
signals are visible to everyone, while resource estimates are noised when the
current region is outside the agent's home region.

## Observation And Action Schema

The base observation includes scenario, task goal, region, resource supplies,
average health/energy, cooperation index, threat level, market volatility,
event severity, region danger, and defense readiness. Hidden verifier fields
exist only in the dataset and are excluded from agent-visible observations.

For the OpenEnv entrypoint, the action schema is a nested multi-agent action
map. `MultiAgentValleyEnv.step()` expects a dictionary with exactly these
agent IDs:

- `farmer`
- `miner`
- `builder`
- `warrior`

Each value is a composite single-agent action:

- `primary_action`
- `focus_resource`
- `cooperation_mode`
- `risk_posture`
- `rationale`

Unknown agent IDs are rejected, and all four required agents must submit an
action on each step. The single-agent `AgentValleyEnv` remains available for
baseline evaluation and compatibility with the tabular Q-learning, neural
policy, and compact GRPO-style trainers.

`env/action_space.py` maps the full finite action tuple to a discrete index so
tabular and neural learners optimize the real decision, not just the primary
action.

## Multi-Agent Mechanics

All four agents submit actions at the same tick. The wrapper uses a lead-agent
rule to keep compatibility with the inner `AgentValleyEnv.step()`:

- high threat: warrior leads
- low food: farmer leads
- low defense readiness: builder leads
- otherwise: lead rotates among farmer, miner, builder, warrior

The lead action advances the inner benchmark and receives the base environment
reward. Every agent also receives role reward, shared cooperation bonus, and
shared conflict penalty. This makes cooperation measurable without breaking the
existing single-agent training code.

## Reward Design

The base environment reward has dense components:

- progress reward
- objective bonus
- focus bonus
- cooperation/risk alignment
- safety penalty
- loop penalty
- format penalty

The multi-agent wrapper adds:

- role reward for role-appropriate action, resource, and cooperation mode
- cooperation bonus for complementary division of labor
- conflict penalty for pile-on, unsafe defense choices, or ignoring starvation

The reward is deterministic from visible state and submitted actions. No random
reward or fake metric is used for training evidence.

## Anti-Reward-Hacking Protections

`env/anti_cheat.py` validates action schemas, rejects extra fields, blocks
hidden-answer leakage terms in rationales, detects unsafe actions under high
threat or low capacity, and flags repeated identical actions. Dataset hidden
targets are never returned by `Observation.agent_view()`.

## Training Methods

`training/q_learning.py` implements real tabular Q-learning with epsilon-greedy
exploration, epsilon decay, real environment rewards, and Bellman updates.

`training/train_neural_policy.py` trains a small CPU PyTorch categorical policy
with policy-gradient updates and saves `artifacts/neural_policy/policy.pt`.

`training/grpo_train.py` implements compact GRPO-style grouped reward
optimization: sampled candidate groups, real replayed environment scoring,
group-relative advantages, clipped log-probability objective, KL, entropy, and
`optimizer.step()`.

`training/ma_grpo_train.py` extends the idea to the multi-agent wrapper with
one small policy per role. It logs team reward, cooperation rate, per-agent
role rewards, per-agent losses, and per-agent checkpoints.

These are CPU-safe RL trainers. They are not full LLM fine-tuning with
TRL/Unsloth.

## Frontend-Backend Architecture

FastAPI in `server/app.py` owns environment state, training jobs, metrics, and
policy evaluation. The React frontend polls backend endpoints for training
metrics and clearly labels the animated canvas as a visual simulation.

The main training dashboard reads:

- `/api/training/status`
- `/api/training/metrics`
- `/api/training/latest`

The multi-agent backend can be inspected through:

- `/api/multi-agent/reset`
- `/api/multi-agent/state`
- `/api/multi-agent/step`
- `/api/multi-agent/evaluate`

## Learning Curves And Evidence

`scripts/generate_training_plots.py` reads backend JSONL files from
`artifacts/` and writes:

- `assets/reward_curve.png`
- `assets/loss_curve.png`
- `assets/training_summary.json`

The reward curve uses real `total_reward` values. The loss curve uses real
`policy_loss` values where available. Q-learning loss is not invented.


## Results Summary

The merged submission keeps the cleaner OpenEnv-compatible multi-agent schema
from `agent-valley` and imports the stronger multi-agent reward evidence from
`final-expanded` under the standardized `multi_agent_grpo` artifact path.

Committed evidence highlights:

- Best overall backend reward: **29.42**
- Latest multi-agent team reward: **27.69**
- Best multi-agent task score: **0.9175**
- Best observed final label: **`valley_stabilized`**
- Multi-agent cooperation rate: **1.0** in the committed MA-GRPO run
- Artifacts: `artifacts/multi_agent_grpo/metrics.jsonl` and per-agent checkpoints
  under `artifacts/multi_agent_grpo/agent_*/policy.pt`

This gives judges an observable before/after-style story: baseline evaluation
still collapses on short deterministic runs, while the coordinated multi-agent
GRPO run reaches much higher team rewards and includes a stabilized episode.

## Known Limitations

- The policy networks are small and CPU-friendly.
- The GRPO implementations are compact optimizer loops, not transformer-scale
  TRL/Unsloth training.
- Episodes are deterministic benchmark tasks rather than an unbounded
  simulation world.
- The React animation is a visual demo separate from backend RL.
- Public Hugging Face Space, GitHub, and video/slides links are still TODO.

## Future Work

Future extensions could connect the same JSON action schema to TRL/Unsloth,
add LoRA adapters for a small instruction model, expand datasets, add
negotiation messages between agents, introduce adversarial resource claims, and
publish a hosted Hugging Face Space with a mini-blog or demo video.
