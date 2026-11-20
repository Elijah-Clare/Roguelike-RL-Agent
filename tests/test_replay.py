# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for replaying runs, the thinking arrows, and saving GIFs.

import glob

import numpy as np
from PIL import Image

from roguelike.replay import (load_demo, random_policy, replay_states, run_frames, run_policy,
                              save_gif)


def fixed_policy(probs):
    """policy that always reports the same probabilities and plays the most likely action"""
    probs = np.array(probs, dtype=np.float32)
    return lambda obs: (int(np.argmax(probs)), probs)


def test_replay_matches_the_original_run():
    """replaying the same actions ends the same way as the original run"""
    actions, probs, result = run_policy(random_policy(1), seed=1000003, run=2)
    *_, (game, _) = replay_states(1000003, 2, actions)
    assert game.done and len(actions) == game.total_turns == len(probs)
    assert result == ("won" if game.won else "died" if game.died else "ran out of turns")


def test_recorded_demos_replay_to_a_win():
    """every recorded human run that says it won also wins when replayed"""
    for path in sorted(glob.glob("demos/*.npz"))[:20]:
        seed, run, floors, actions = load_demo(path)
        *_, (game, _) = replay_states(seed, run, actions, floors)
        assert game.won, path


def test_gif_has_one_frame_per_turn(tmp_path):
    """the GIF shows the start, every turn, and then holds on the last frame"""
    actions, _, _ = run_policy(random_policy(0), seed=1000001, run=0)
    actions = actions[:10]
    frames = run_frames(1000001, 0, actions, "test")
    assert len(frames) == len(actions) + 1
    assert len({frame.tobytes() for frame in frames}) == len(frames)  # turn counter differs each frame
    gif = tmp_path / "run.gif"
    save_gif(frames, gif, fps=5)
    with Image.open(gif) as image:
        assert image.n_frames >= 2


def test_thinking_arrows_change_the_picture():
    """frames with probabilities draw arrows that frames without them do not"""
    policy = fixed_policy([0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
    actions, probs, _ = run_policy(policy, seed=1000002, run=0)
    plain = run_frames(1000002, 0, actions[:3], "plain")
    thinking = run_frames(1000002, 0, actions[:3], "plain", probs[:3])
    assert plain[0].tobytes() != thinking[0].tobytes()
    assert plain[-1].tobytes() == thinking[-1].tobytes()  # no probs left for the last frame
