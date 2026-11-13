# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Gymnasium wrapper around the game so Stable-Baselines3 can train on it.
# The agent sees an 11x11 window centered on itself plus a few stats.

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dungeon import STAIRS, WALL
from .game import NUM_ACTIONS, Game

VIEW_SIZE = 11

# channels of the view, in order
WALL_CH, STAIRS_CH, VISITED_CH, ENEMY_CH, POTION_CH = range(5)
NUM_CHANNELS = 5


def outcome_reward(events):
    """placeholder reward until reward.py is ready: +10 for the stairs only"""
    return 10.0 if events["reached_stairs"] else 0.0


class RoguelikeEnv(gym.Env):
    """the roguelike as a gymnasium environment"""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, num_floors=1, turns_per_floor=200, seed_pool=None, reward_fn=outcome_reward,
                 render_mode=None, **game_kwargs):
        """seed_pool limits which maps reset() picks from (e.g. only training seeds)"""
        self.num_floors = num_floors
        self.turns_per_floor = turns_per_floor
        self.seed_pool = list(seed_pool) if seed_pool is not None else None
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.game_kwargs = game_kwargs  # extra settings passed to Game
        self.game = None

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "view": spaces.Box(0.0, 1.0, (NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), np.float32),
            "stats": spaces.Box(0.0, 1.0, (3,), np.float32),
        })

    def reset(self, seed=None, options=None):
        """start a new run. reset(seed=s) always gives map s, otherwise one is drawn from seed_pool"""
        super().reset(seed=seed)
        if seed is not None:
            map_seed = seed
        elif self.seed_pool is not None:
            map_seed = self.seed_pool[int(self.np_random.integers(len(self.seed_pool)))]
        else:
            map_seed = int(self.np_random.integers(2**31))
        self.game = Game(map_seed, num_floors=self.num_floors, turns_per_floor=self.turns_per_floor,
                         **self.game_kwargs)
        return self.observe(), {"map_seed": map_seed}

    def step(self, action):
        """play one turn and return obs, reward, terminated, truncated, info"""
        events = self.game.step(int(action)).as_dict()
        reward = float(self.reward_fn(events))
        terminated = self.game.won or self.game.died
        truncated = self.game.timed_out
        info = {"events": events, "floor": self.game.floor, "won": self.game.won,
                "turns": self.game.total_turns}
        return self.observe(), reward, terminated, truncated, info

    def observe(self):
        """the 11x11 view around the player plus hp, floor, and turns used"""
        game = self.game
        half = VIEW_SIZE // 2
        view = np.zeros((NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE), np.float32)
        view[WALL_CH] = 1.0  # outside the map counts as wall

        # copy the part of the map that falls inside the view
        view_top, view_left = game.pos[0] - half, game.pos[1] - half
        top, left = max(view_top, 0), max(view_left, 0)
        bottom = min(view_top + VIEW_SIZE, game.height)
        right = min(view_left + VIEW_SIZE, game.width)
        patch = game.grid[top:bottom, left:right]
        rows = slice(top - view_top, bottom - view_top)
        cols = slice(left - view_left, right - view_left)
        view[WALL_CH, rows, cols] = patch == WALL
        view[STAIRS_CH, rows, cols] = patch == STAIRS
        for row, col in game.visited:
            if top <= row < bottom and left <= col < right:
                view[VISITED_CH, row - view_top, col - view_left] = 1.0
        for enemy in game.enemies:
            row, col = enemy.pos
            if top <= row < bottom and left <= col < right:
                view[ENEMY_CH, row - view_top, col - view_left] = enemy.kind.obs_value

        stats = np.array([game.hp / game.max_hp,
                          game.floor / game.num_floors,
                          game.turns_on_floor / game.turns_per_floor], np.float32)
        return {"view": view, "stats": stats}

    def render(self):
        """text picture of the map when render_mode is "ansi" """
        if self.render_mode == "ansi":
            return self.game.render_text()
