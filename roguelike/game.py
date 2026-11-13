# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Game rules: one player moving through seeded dungeon floors.
# Each call to step() is one turn and returns what happened that turn.

from dataclasses import dataclass, asdict

import numpy as np

from .dungeon import FLOOR, WALL, distances_from, generate_dungeon
from .enemies import ENEMY_TYPES, Enemy, enemy_turn

PLAYER_DAMAGE = 3        # hp an enemy loses when the player bumps it
MIN_ENEMY_DISTANCE = 4   # enemies spawn at least this many steps from the start

# action ids
UP, DOWN, LEFT, RIGHT, WAIT, DRINK = range(6)
NUM_ACTIONS = 6

# (row change, column change) for each move
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


@dataclass
class Events:
    """what happened during one turn (the reward function reads these)"""
    moved: bool = False
    hit_wall: bool = False
    new_tile: bool = False        # stepped on a tile for the first time
    attacked: bool = False        # bumped into an enemy
    killed_enemy: bool = False
    damage_taken: int = 0         # hp lost to enemies this turn
    reached_stairs: bool = False
    won: bool = False             # cleared the last floor
    died: bool = False
    timed_out: bool = False       # hit the turn limit for this floor

    def as_dict(self):
        """events as a plain dict"""
        return asdict(self)


class Game:
    """one run of the game from the first floor until it ends"""

    def __init__(self, seed, behavior_seed=None, num_floors=1, width=12, height=12,
                 turns_per_floor=200, max_hp=10):
        """start a new run. seed decides the maps, behavior_seed decides how enemies move"""
        self.rng = np.random.default_rng(seed)  # maps and spawns
        self.behavior_rng = np.random.default_rng(seed if behavior_seed is None else behavior_seed)
        self.num_floors = num_floors
        self.width = width
        self.height = height
        self.turns_per_floor = turns_per_floor
        self.max_hp = max_hp
        self.hp = max_hp
        self.floor = 1
        self.total_turns = 0
        self.won = False
        self.died = False
        self.timed_out = False
        self.new_floor()

    def new_floor(self):
        """build the next floor and put the player at its start"""
        self.dungeon = generate_dungeon(self.rng, self.width, self.height)
        self.pos = self.dungeon.start
        self.visited = {self.pos}  # tiles the player has stood on this floor
        self.turns_on_floor = 0
        self.enemies = self.spawn_enemies()

    def spawn_enemies(self):
        """place 1 to 3 enemies far enough from the start"""
        dist = distances_from(self.grid, self.pos)
        tiles = sorted(tile for tile, d in dist.items()
                       if d >= MIN_ENEMY_DISTANCE and self.grid[tile] == FLOOR)
        count = min(int(self.rng.integers(1, 4)), len(tiles))
        picks = self.rng.choice(len(tiles), size=count, replace=False)
        enemies = []
        for index in picks:
            kind = ENEMY_TYPES[int(self.rng.integers(len(ENEMY_TYPES)))]
            enemies.append(Enemy(kind, tiles[int(index)], kind.max_hp))
        return enemies

    def enemy_at(self, tile):
        """the enemy standing on tile, or None"""
        for enemy in self.enemies:
            if enemy.pos == tile:
                return enemy
        return None

    @property
    def grid(self):
        """tile grid of the current floor"""
        return self.dungeon.grid

    @property
    def done(self):
        """true once the run has ended"""
        return self.won or self.died or self.timed_out

    def step(self, action):
        """play one turn and return the Events for it"""
        if self.done:
            raise RuntimeError("game is over, start a new one")
        events = Events()

        if action in MOVES:
            d_row, d_col = MOVES[action]
            target = (self.pos[0] + d_row, self.pos[1] + d_col)
            enemy = self.enemy_at(target)
            if enemy:
                events.attacked = True
                enemy.hp -= PLAYER_DAMAGE
                if enemy.hp <= 0:
                    self.enemies.remove(enemy)
                    events.killed_enemy = True
            elif self.grid[target] == WALL:
                events.hit_wall = True
            else:
                self.pos = target
                events.moved = True
                if target not in self.visited:
                    self.visited.add(target)
                    events.new_tile = True
        # WAIT does nothing. DRINK does nothing until potions are added

        self.total_turns += 1
        self.turns_on_floor += 1

        if self.pos == self.dungeon.stairs:
            events.reached_stairs = True
            if self.floor == self.num_floors:
                self.won = events.won = True
            else:
                self.floor += 1
                self.new_floor()
            return events

        # enemies take their turns after the player
        for enemy in self.enemies:
            damage = enemy_turn(enemy, self, self.behavior_rng)
            self.hp -= damage
            events.damage_taken += damage
            if self.hp <= 0:
                self.hp = 0
                self.died = events.died = True
                return events

        if self.turns_on_floor >= self.turns_per_floor:
            self.timed_out = events.timed_out = True
        return events

    def render_text(self):
        """the map as text (# wall, . floor, > stairs, @ player, r/g enemies) plus a status line"""
        chars = [["#.>"[tile] for tile in tiles] for tiles in self.grid]
        for enemy in self.enemies:
            chars[enemy.pos[0]][enemy.pos[1]] = enemy.kind.symbol
        chars[self.pos[0]][self.pos[1]] = "@"
        rows = ["".join(line) for line in chars]
        status = (f"Floor {self.floor}/{self.num_floors}  HP {self.hp}/{self.max_hp}  "
                  f"Turn {self.turns_on_floor}/{self.turns_per_floor}")
        return "\n".join(rows + [status])
