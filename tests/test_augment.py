# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for turning and mirroring recorded runs.

import numpy as np

from roguelike.demos import augment, transform_action, transform_view
from roguelike.env import VIEW_SIZE, WALL_CH, RoguelikeEnv
from roguelike.game import DOWN, DRINK, LEFT, MOVES, RIGHT, UP, WAIT

TRANSFORMS = [(rotations, flip) for rotations in range(4) for flip in (False, True)]


def test_identity_changes_nothing():
    """no turn and no mirror leaves the view and actions alone"""
    view = np.random.default_rng(0).random((5, VIEW_SIZE, VIEW_SIZE))
    assert np.array_equal(transform_view(view, 0, False), view)
    for action in range(6):
        assert transform_action(action, 0, False) == action


def test_known_moves():
    """a quarter turn counterclockwise makes up into left, and a mirror swaps left and right"""
    assert transform_action(UP, 1, False) == LEFT
    assert transform_action(RIGHT, 1, False) == UP
    assert transform_action(LEFT, 0, True) == RIGHT
    assert transform_action(UP, 0, True) == UP
    assert transform_action(WAIT, 1, True) == WAIT
    assert transform_action(DRINK, 3, True) == DRINK


def test_moves_still_point_at_the_same_tile():
    """after a transform, the new move still leads to the same kind of tile (wall or open)"""
    env = RoguelikeEnv()
    half = VIEW_SIZE // 2
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        walls = obs["view"][WALL_CH]
        for rotations, flip in TRANSFORMS:
            new_walls = transform_view(walls, rotations, flip)
            for action in MOVES:
                d_row, d_col = MOVES[action]
                n_row, n_col = MOVES[transform_action(action, rotations, flip)]
                assert walls[half + d_row, half + d_col] == new_walls[half + n_row, half + n_col]


def test_all_eight_versions_differ():
    """the 8 turns and mirrors of a lopsided view are all different"""
    view = np.zeros((1, VIEW_SIZE, VIEW_SIZE))
    view[0, 0, 1] = 1  # one mark off the diagonals
    versions = {transform_view(view, r, f).tobytes() for r, f in TRANSFORMS}
    assert len(versions) == 8


def test_augment_makes_eight_times_the_data():
    """augment returns 8 copies with stats unchanged and actions remapped"""
    demos = {"view": np.random.default_rng(1).random((3, 5, VIEW_SIZE, VIEW_SIZE)),
             "stats": np.ones((3, 4)), "actions": np.array([UP, DOWN, WAIT]), "runs": 1}
    big = augment(demos)
    assert big["view"].shape == (24, 5, VIEW_SIZE, VIEW_SIZE)
    assert big["stats"].shape == (24, 4) and len(big["actions"]) == 24
    assert list(big["actions"][:3]) == [UP, DOWN, WAIT]
    assert (big["actions"][2::3] == WAIT).all()
