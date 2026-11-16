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

## Playing the game
```
python -m roguelike.play
```
Arrow keys or WASD move (and attack), space waits, Q drinks a potion, R starts a new run, Esc quits.
The window uses the 16x16 pixel art in `assets/` (floor, wall, stairs, player, rat, goblin, potion, heart). If a file is missing it draws a plain shape instead. The art only changes how the window looks. The agent never sees it.
To record runs for the imitation baseline:
```
python -m roguelike.play --record --name dashawn
```
Each finished run (win, death, or timeout) is saved to `demos/` as an `.npz` file. Pressing R before a run ends skips saving it. `roguelike.demos.load_demos()` loads every recorded step as `view`, `stats`, and `actions` arrays. `roguelike.demos.augment(demos)` adds every quarter turn and mirror image of each step, with the actions remapped to match, giving 8 times the data.

Human play only uses training maps. Seeds are split in `roguelike/seeds.py`: training maps are 0 to 9999, test maps start at 1,000,000.

## Game rules
- 12x12 dungeon of rooms and corridors. Reach the stairs (`>`) to clear the floor.
- Player has 10 HP. Moving into an enemy attacks it for 3 damage.
- Enemies (1 to 3 per floor, they do not respawn):
  - rat `r`: 3 HP, hits for 1, chases when within 4 steps
  - goblin `g`: 6 HP, hits for 2, chases when within 6 steps
  - next to the player they always attack. When chasing, they move randomly 20% of the time. Otherwise they wander.
- Potions `!` (1 or 2 per floor): walk over one to pick it up (carry up to 3). Drinking heals 5 HP, capped at max HP.
- Turn order: the player acts, then each enemy acts.
- Turn limit: 200 turns per floor.

## Using the environment
```python
from roguelike.env import RoguelikeEnv

env = RoguelikeEnv(seed_pool=range(0, 10000))      # training maps
obs, info = env.reset()                              # random map from the pool
obs, info = env.reset(seed=1_000_000, options={"run": 2})  # test map 1000000, run 2
obs, reward, terminated, truncated, info = env.step(action)
```

`seed` picks the map. `options={"run": r}` picks how the enemies move, so the 5 runs on one test map should pass run 0 to 4. The same seed and run always replay exactly the same way.

**Actions** (`Discrete(6)`): 0 up, 1 down, 2 left, 3 right, 4 wait, 5 drink potion. Moving into an enemy attacks it.

**Observation** (`Dict`, use `MultiInputPolicy` in Stable-Baselines3):
- `view`: shape (5, 11, 11), centered on the player. Channels: 0 wall (outside the map counts as wall), 1 stairs, 2 tiles already visited, 3 enemies (0.5 rat, 1.0 goblin), 4 potions.
- `stats`: health / max health, floor / number of floors, turns used on this floor / turn limit, potions carried / 3.

**Episode end:** `terminated` when the agent clears the last floor or dies. `truncated` when it hits the turn limit.

**Rewards:** the env calls `reward_fn(events)` each step. The default only gives +10 for reaching the stairs. Pass your own function with `RoguelikeEnv(reward_fn=...)`. `events` is also returned in `info["events"]` and has:
- `moved`, `hit_wall`, `new_tile` (first time on that tile)
- `attacked`, `killed_enemy`, `damage_taken` (HP lost this turn)
- `picked_potion`, `drank_potion`, `healed` (HP restored, 0 if already full)
- `reached_stairs`, `won`, `died`, `timed_out`
