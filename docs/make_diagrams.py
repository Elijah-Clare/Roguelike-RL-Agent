# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Draws the game loop and project workflow diagrams used in the report.
# Usage: python docs/make_diagrams.py   (saves PNGs to docs/figures/)

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLUE = "#dce8f7"
GREEN = "#dff0dc"
YELLOW = "#fbf0cf"
GRAY = "#ececec"
EDGE = "#3c3c3c"


def box(ax, x, y, w, h, text, color, size=10, bold=False):
    """rounded box centered at (x, y) with text inside"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor=EDGE, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, start, end, text=None, curve=0.0):
    """arrow from start to end with an optional label in the middle"""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.3, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={curve}"))
    if text:
        ax.text((start[0] + end[0]) / 2 + 0.08, (start[1] + end[1]) / 2, text, fontsize=9,
                ha="left", va="center", style="italic")


def game_loop():
    """one turn of the game from the agent's point of view, plus where PPO learns"""
    fig, ax = plt.subplots(figsize=(7, 8.2))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    x, w, h = 2.7, 3.6, 0.62
    steps = [
        (7.6, "Start a run\n(map seed + run number)", GRAY),
        (6.6, "Agent observes\n11x11 view + HP, floor, turns, potions", BLUE),
        (5.6, "Policy picks an action\n(move/attack, wait, or drink)", BLUE),
        (4.6, "Game applies the player's action", GREEN),
        (3.6, "Each enemy takes its turn\n(attack, chase, or wander)", GREEN),
        (2.6, "Reward from this turn's events", YELLOW),
        (1.5, "Run over?\n(won, died, or turn limit)", GRAY),
    ]
    for y, text, color in steps:
        box(ax, x, y, w, h, text, color)
    for (y1, _, _), (y2, _, _) in zip(steps, steps[1:]):
        arrow(ax, (x, y1 - h / 2), (x, y2 + h / 2))

    box(ax, x, 0.45, 1.8, 0.5, "End run", GRAY, bold=True)
    arrow(ax, (x, 1.5 - h / 2), (x, 0.45 + 0.25), "yes")

    # loop back to observing when the run is not over
    loop_x = x - w / 2 - 0.35
    ax.plot([x - w / 2, loop_x, loop_x], [1.5, 1.5, 6.6], color=EDGE, lw=1.3)
    arrow(ax, (loop_x, 6.6), (x - w / 2, 6.6))
    ax.text(loop_x - 0.08, 4.0, "no, next turn", rotation=90, fontsize=9, ha="right", va="center",
            style="italic")

    # where learning happens
    box(ax, 5.9, 4.1, 1.85, 1.5, "PPO update\nafter every 2,048\nturns collected", YELLOW, bold=True)
    arrow(ax, (x + w / 2, 2.6), (5.9, 4.1 - 0.75), curve=0.25)
    arrow(ax, (5.9, 4.1 + 0.75), (x + w / 2, 5.6), curve=0.25)
    ax.text(6.15, 2.75, "rewards", fontsize=9, style="italic")
    ax.text(6.15, 5.35, "better policy", fontsize=9, style="italic")

    ax.set_title("Game loop", fontsize=13, weight="bold")
    return fig


def workflow():
    """the whole project from the game to the final evaluation"""
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 5.2)
    ax.axis("off")

    stages = [
        (1.0, "1. Game", GREEN,
         "Seeded dungeon\ngenerator\n\nRules: enemies,\npotions, stairs,\nturn limit\n\nGymnasium env\n(11x11 view)"),
        (3.0, "2. Data", GRAY,
         "Training maps:\nseeds 0-9,999\n\nTest maps:\nseeds 1,000,000+\n\n101 human runs\nx8 by rotate/flip"),
        (5.0, "3. Agents", BLUE,
         "Random\n\nScripted bot\n\nImitation\n(copies human runs)\n\nPPO (sparse and\nshaped reward)"),
        (7.0, "4. Training", YELLOW,
         "Stable-Baselines3\nPPO\n\nreward.py with\nexplore decay\n\nCheckpoints\nduring training"),
        (9.0, "5. Evaluation", GRAY,
         "100 test maps\nx 5 runs each\n\nWin rate, floors,\nturns, damage\n\nLearning curves\nand replays"),
    ]
    for x, title, color, body in stages:
        box(ax, x, 4.45, 1.75, 0.65, title, color, size=10.5, bold=True)
        box(ax, x, 2.6, 1.75, 2.75, body, "white", size=9)
    for (x1, *_), (x2, *_) in zip(stages, stages[1:]):
        arrow(ax, (x1 + 0.875, 4.45), (x2 - 0.875, 4.45))

    ax.text(5.0, 0.75, "Tools: Python, Gymnasium, pygame, Stable-Baselines3 (PyTorch), NumPy, Matplotlib, pytest",
            ha="center", fontsize=9.5)
    ax.set_title("Project workflow", fontsize=13, weight="bold")
    return fig


def main():
    """save both diagrams as PNGs"""
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, make in (("game_loop", game_loop), ("workflow", workflow)):
        fig = make()
        path = os.path.join(OUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
