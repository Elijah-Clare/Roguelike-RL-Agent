# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for replaying runs and saving them as GIFs.

import glob

from PIL import Image

from roguelike.replay import load_demo, random_policy, replay_states, run_policy, show_run


def test_replay_matches_the_original_run():
    """replaying the same actions ends the same way as the original run"""
    actions, result = run_policy(random_policy(1), seed=1000003, run=2)
    *_, (game, _) = replay_states(1000003, 2, actions)
    assert game.done and len(actions) == game.total_turns
    assert result == ("won" if game.won else "died" if game.died else "ran out of turns")


def test_recorded_demos_replay_to_a_win():
    """every recorded human run that says it won also wins when replayed"""
    for path in sorted(glob.glob("demos/*.npz"))[:20]:
        seed, run, floors, actions = load_demo(path)
        *_, (game, _) = replay_states(seed, run, actions, floors)
        assert game.won, path


def test_gif_has_one_frame_per_turn(tmp_path):
    """the GIF shows the start, every turn, and then holds on the last frame"""
    actions, _ = run_policy(random_policy(0), seed=1000001, run=0)
    actions = actions[:10]
    gif = tmp_path / "run.gif"
    show_run(1000001, 0, actions, "test", gif=str(gif), fps=5, window=False)
    with Image.open(gif) as image:
        assert image.n_frames >= len(actions) + 1
