# PROGRAMMER: Elijah Clare
# DUE DATE: Monday, 11/30/26
# Evaluate all PPO checkpoints on held-out test seeds and plot win rate.

import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from evaluation.evaluate import evaluate, summarize


def extract_step(path: Path) -> int:
    # Grab the final number in the filename (the training step count)
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else 0


def main() -> None:
    # Find and sort all saved checkpoints from lowest to highest step
    ckpts = sorted(Path("checkpoints").glob("*_step_*.zip"), key=extract_step)
    if not ckpts:
        raise FileNotFoundError("No checkpoints/*_step_*.zip files found.")

    Path("results").mkdir(exist_ok=True)
    rows = []

    # Score each checkpoint on 500 unseen test runs (100 seeds x 5 runs)
    for ckpt in ckpts:
        step = extract_step(ckpt)
        agent = PPOAgent(str(ckpt))
        results = evaluate(agent, num_seeds=100, runs_per_seed=5)
        summary = summarize(results)
        summary["step"] = step
        rows.append(summary)
        print(
            f"step={step:>8} | win_rate={summary['won']:.3f} "
            f"| floor={summary['floor']:.2f} | turns={summary['turns']:.1f}"
        )

    # Save summary metrics to CSV for the report table
    csv_path = Path("results/ppo_learning_curve.csv")
    fieldnames = ["step", "won", "died", "floor", "turns", "new_tiles", "reward"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Plot held-out win rate (%) vs. training steps
    steps = [r["step"] for r in rows]
    win_rates = [r["won"] * 100 for r in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, win_rates, marker="o")
    plt.xlabel("Training Steps")
    plt.ylabel("Held-Out Win Rate (%)")
    plt.title("PPO Held-Out Win Rate vs. Training Steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150)
    print("Saved results/ppo_learning_curve.csv and results/learning_curve.png")


if __name__ == "__main__":
    main()