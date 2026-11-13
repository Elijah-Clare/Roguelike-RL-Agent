# Roguelike-RL-Agent
Reinforcement learning agent for a roguelike game. This project is being done for Intro to AI CSCI-4350 Fall 2026.

Group members:
Elijah Clare
Dashawn Williams

## Setup
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Using the environment
```python
from roguelike.env import RoguelikeEnv

env = RoguelikeEnv(seed_pool=range(0, 10000))  # training maps
obs, info = env.reset()                          # random map from the pool
obs, info = env.reset(seed=1_000_000)            # a specific map, e.g. a test seed
obs, reward, terminated, truncated, info = env.step(action)
```

**Actions** (`Discrete(6)`): 0 up, 1 down, 2 left, 3 right, 4 wait, 5 drink potion.
Moving into an enemy will attack it once enemies are added.

**Observation** (`Dict`, use `MultiInputPolicy` in Stable-Baselines3):
- `view`: shape (5, 11, 11), centered on the player. Channels: 0 wall (outside the map counts as wall), 1 stairs, 2 tiles already visited, 3 enemies, 4 potions. Enemies and potions are empty until they are added.
- `stats`: health / max health, floor / number of floors, turns used on this floor / turn limit.

**Episode end:** `terminated` when the agent clears the last floor (or dies, once enemies exist). `truncated` when it hits the turn limit (200 turns per floor by default).

**Rewards:** the env calls `reward_fn(events)` each step. The default only gives +10 for reaching the stairs. Pass your own function with `RoguelikeEnv(reward_fn=...)`. `events` is a dict with `moved`, `hit_wall`, `new_tile`, `reached_stairs`, `won`, `died`, `timed_out`. It is also returned in `info["events"]`.
