# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for the side by side comparison GIF.

from roguelike.compare import compare_frames
from roguelike.replay import save_gif


def test_panels_line_up_in_a_grid(tmp_path):
    """four agents make a 2 by 2 grid with one frame per turn of the longest run"""
    frames, results = compare_frames(1000003, 0, ["random"] * 4, ["a", "b", "c", "d"], max_turns=8)
    assert len(results) == 4
    longest = max(turns for _, _, turns in results)
    assert len(frames) == longest + 1
    one_wide = compare_frames(1000003, 0, ["random"], ["a"], max_turns=8)[0][0].width
    assert frames[0].width > 2 * one_wide - 10
    gif = tmp_path / "compare.gif"
    save_gif(frames, gif, fps=5)
    assert gif.stat().st_size < 5_000_000


def test_max_turns_cuts_long_runs():
    """runs longer than max_turns stop early and say they were still going"""
    _, results = compare_frames(1000000, 0, ["random"], ["a"], max_turns=3)
    label, result, turns = results[0]
    assert turns <= 3
    assert result in ("still going", "won", "died")
