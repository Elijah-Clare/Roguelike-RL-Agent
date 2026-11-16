# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Saving and loading recorded human runs for the imitation baseline.
# Each run is one .npz file with what the player saw and what they pressed.

import glob
import os
from datetime import datetime

import numpy as np

from .game import MOVES, NUM_ACTIONS

DEMO_DIR = "demos"


class Recorder:
    """collects one run's observations and actions so it can be saved"""

    def __init__(self, player, map_seed, run, num_floors):
        """start an empty recording for one run"""
        self.player = player
        self.map_seed = map_seed
        self.run = run
        self.num_floors = num_floors
        self.views, self.stats, self.actions = [], [], []

    def add(self, obs, action):
        """store what the player saw and the action they picked"""
        self.views.append(obs["view"])
        self.stats.append(obs["stats"])
        self.actions.append(action)

    def save(self, won, died, folder=DEMO_DIR):
        """write the run to an .npz file and return its path"""
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"{self.player}_{stamp}_seed{self.map_seed}.npz")
        np.savez_compressed(path, view=np.array(self.views, np.float32),
                            stats=np.array(self.stats, np.float32),
                            actions=np.array(self.actions, np.int64),
                            map_seed=self.map_seed, run=self.run, num_floors=self.num_floors,
                            won=won, died=died, player=self.player)
        return path


def load_demos(folder=DEMO_DIR, only_won=False):
    """every recorded step stacked together as view, stats, and actions arrays"""
    views, stats, actions = [], [], []
    runs = 0
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        with np.load(path) as demo:
            if only_won and not demo["won"]:
                continue
            views.append(demo["view"])
            stats.append(demo["stats"])
            actions.append(demo["actions"])
            runs += 1
    if not runs:
        raise FileNotFoundError(f"no recorded runs in {folder}")
    return {"view": np.concatenate(views), "stats": np.concatenate(stats),
            "actions": np.concatenate(actions), "runs": runs}


def transform_view(view, rotations, flip):
    """turn the view a quarter turn rotations times, then mirror it left to right if flip"""
    out = np.rot90(view, k=rotations, axes=(-2, -1))
    if flip:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def transform_action(action, rotations, flip):
    """the move that matches action after the same turn and mirror (wait and drink stay the same)"""
    if action not in MOVES:
        return action
    d_row, d_col = MOVES[action]
    marker = np.zeros((3, 3))
    marker[1 + d_row, 1 + d_col] = 1  # mark the tile the move goes to
    row, col = np.argwhere(transform_view(marker, rotations, flip))[0]
    new_move = (int(row) - 1, int(col) - 1)
    return next(other for other, move in MOVES.items() if move == new_move)


def augment(demos):
    """all 8 turns and mirror images of the recorded steps (8 times the data)"""
    views, stats, actions = [], [], []
    for rotations in range(4):
        for flip in (False, True):
            table = np.array([transform_action(a, rotations, flip) for a in range(NUM_ACTIONS)])
            views.append(transform_view(demos["view"], rotations, flip))
            stats.append(demos["stats"])
            actions.append(table[demos["actions"]])
    return {"view": np.concatenate(views), "stats": np.concatenate(stats),
            "actions": np.concatenate(actions), "runs": demos["runs"]}
