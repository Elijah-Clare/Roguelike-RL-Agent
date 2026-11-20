# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Side by side GIF of several agents or checkpoints playing the same map with the same enemy moves.
# Made to show the agent learning: before training, early, middle, and final on one test map.
# Usage:
#   python -m roguelike.compare --seed 1000003 --run 0 --gif learning.gif
#       --agents random checkpoints/ppo_step_50000.zip checkpoints/ppo_step_200000.zip checkpoints/ppo_final.zip
#       --labels "No training" "50k steps" "200k steps" "Final"

import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont

from .replay import model_policy, random_policy, run_frames, run_policy

HEADER = 34  # pixels for the label above each panel
GAP = 6      # pixels between panels
RESULT_COLORS = {"won": (40, 160, 70), "died": (190, 45, 45), "ran out of turns": (110, 110, 110),
                 "still going": (60, 90, 150)}


def load_agent(spec):
    """policy and a default label for "random" or a saved model path"""
    if spec == "random":
        return random_policy(), "Random"
    return model_policy(spec), os.path.splitext(os.path.basename(spec))[0]


def font(size):
    """a bold font for labels, or Pillow's built in one if consolas is missing"""
    try:
        return ImageFont.truetype("consolab.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def add_banner(frame, result, turns):
    """write how the run ended across the middle of a panel"""
    frame = frame.convert("RGBA")
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    text = {"won": f"CLEARED in {turns} turns", "died": f"DIED on turn {turns}",
            "ran out of turns": "OUT OF TURNS", "still going": f"STILL GOING at turn {turns}"}[result]
    top = frame.height // 2 - 30
    draw.rectangle((0, top, frame.width, top + 60), fill=(*RESULT_COLORS[result], 215))
    draw.text((frame.width // 2, top + 30), text, font=font(30), fill="white", anchor="mm")
    return Image.alpha_composite(frame, layer).convert("RGB")


def add_header(frame, label):
    """put a big label above a panel"""
    out = Image.new("RGB", (frame.width, frame.height + HEADER), (20, 20, 24))
    out.paste(frame, (0, HEADER))
    ImageDraw.Draw(out).text((frame.width // 2, HEADER // 2), label, font=font(22), fill="white", anchor="mm")
    return out


def compare_frames(seed, run, specs, labels, num_floors=1, thinking=True, scale=0.6, columns=None,
                   max_turns=None):
    """one combined frame per turn with every agent side by side. also returns each result"""
    panels, results = [], []
    for spec, label in zip(specs, labels):
        policy, _ = load_agent(spec)
        actions, probs, result = run_policy(policy, seed, run, num_floors)
        if max_turns and len(actions) > max_turns:  # cut long runs short for slides
            actions, probs, result = actions[:max_turns], probs[:max_turns], "still going"
        frames = run_frames(seed, run, actions, f"{label}  seed {seed}", probs if thinking else None,
                            num_floors)
        frames[-1] = add_banner(frames[-1], result, len(actions))
        size = (int(frames[0].width * scale), int(frames[0].height * scale))
        panels.append([add_header(f.resize(size, Image.NEAREST), label) for f in frames])  # keeps pixels sharp
        results.append((label, result, len(actions)))

    length = max(len(frames) for frames in panels)
    panels = [frames + [frames[-1]] * (length - len(frames)) for frames in panels]  # finished runs wait

    columns = columns or (2 if len(panels) == 4 else len(panels))
    rows = math.ceil(len(panels) / columns)
    width, height = panels[0][0].size
    combined = []
    for turn in range(length):
        sheet = Image.new("RGB", (columns * width + (columns - 1) * GAP, rows * height + (rows - 1) * GAP),
                          (10, 10, 12))
        for i, frames in enumerate(panels):
            sheet.paste(frames[turn], ((i % columns) * (width + GAP), (i // columns) * (height + GAP)))
        combined.append(sheet)
    return combined, results


def main():
    """build the side by side GIF from the command line"""
    from .replay import save_gif

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="map seed (use a test seed)")
    parser.add_argument("--run", type=int, default=0, help="enemy moves, the same for every agent")
    parser.add_argument("--agents", nargs="+", required=True, help='"random" or saved model paths')
    parser.add_argument("--labels", nargs="+", help="one label per agent")
    parser.add_argument("--gif", required=True)
    parser.add_argument("--floors", type=int, default=1)
    parser.add_argument("--columns", type=int, help="panels per row (default 2 for four agents)")
    parser.add_argument("--scale", type=float, default=0.6, help="size of each panel")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--no-thinking", action="store_true", help="hide the PPO move arrows")
    parser.add_argument("--max-turns", type=int, help="stop each run after this many turns")
    args = parser.parse_args()

    labels = args.labels or [load_agent(spec)[1] for spec in args.agents]
    if len(labels) != len(args.agents):
        parser.error("give one label per agent")
    frames, results = compare_frames(args.seed, args.run, args.agents, labels, args.floors,
                                     not args.no_thinking, args.scale, args.columns, args.max_turns)
    save_gif(frames, args.gif, args.fps)
    for label, result, turns in results:
        print(f"{label}: {result} in {turns} turns")
    print(f"saved {args.gif}")


if __name__ == "__main__":
    main()
