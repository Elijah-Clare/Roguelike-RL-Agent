# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for recording and loading human runs.

import numpy as np
import pytest

from roguelike.demos import Recorder, load_demos
from roguelike.env import RoguelikeEnv


def record_run(folder, seed, run, steps=25):
    """play random actions on one map and save the recording"""
    env = RoguelikeEnv()
    obs, _ = env.reset(seed=seed, options={"run": run})
    recorder = Recorder("tester", seed, run, num_floors=env.num_floors)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        action = int(rng.integers(6))
        recorder.add(obs, action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return recorder.save(won=env.game.won, died=env.game.died, folder=folder), len(recorder.actions)


def test_save_and_load(tmp_path):
    """saved runs load back with one row per step"""
    _, steps_a = record_run(tmp_path, seed=1, run=0)
    _, steps_b = record_run(tmp_path, seed=2, run=0)
    demos = load_demos(tmp_path)
    assert demos["runs"] == 2
    assert len(demos["actions"]) == steps_a + steps_b
    assert demos["view"].shape == (steps_a + steps_b, 5, 11, 11)
    assert demos["stats"].shape == (steps_a + steps_b, 4)


def test_recording_replays_exactly(tmp_path):
    """the saved seed, run, and actions rebuild the same observations"""
    path, _ = record_run(tmp_path, seed=5, run=3)
    with np.load(path) as demo:
        env = RoguelikeEnv()
        obs, _ = env.reset(seed=int(demo["map_seed"]), options={"run": int(demo["run"])})
        for view, action in zip(demo["view"], demo["actions"]):
            assert np.array_equal(obs["view"], view)
            obs, *_ = env.step(int(action))


def test_empty_folder_raises(tmp_path):
    """loading from a folder with no runs is an error"""
    with pytest.raises(FileNotFoundError):
        load_demos(tmp_path)
