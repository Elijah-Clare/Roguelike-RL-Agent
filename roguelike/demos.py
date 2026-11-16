# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Saving and loading recorded human runs for the imitation baseline.
# Each run is one .npz file with what the player saw and what they pressed.

import glob
import os
from datetime import datetime

import numpy as np

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
