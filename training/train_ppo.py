# PROGRAMMER: Elijah Clare
# DUE DATE: Monday, 11/30/26
# Train a PPO agent on RoguelikeEnv using Stable-Baselines3.

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from roguelike.env import RoguelikeEnv
from roguelike.seeds import TRAIN_SEEDS
from roguelike.reward import ShapedReward, outcome_reward

class DecayAndEvalCallback(BaseCallback):
    def __init__(self, reward_fn, total_timesteps: int, eval_freq: int = 50_000):
        super().__init__()
        self.reward_fn = reward_fn
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if hasattr(self.reward_fn, "set_progress"):
            self.reward_fn.set_progress(self.num_timesteps / self.total_timesteps)
        if self.n_calls % self.eval_freq == 0:
            self.model.save(f"checkpoints/ppo_step_{self.num_timesteps}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["sparse", "shaped"], default="sparse")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--out", default="checkpoints/ppo_sparse")
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    reward_fn = outcome_reward if args.reward == "sparse" else ShapedReward()
    env = Monitor(RoguelikeEnv(seed_pool=TRAIN_SEEDS, reward_fn=reward_fn))

    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="logs/")
    callback = DecayAndEvalCallback(reward_fn, total_timesteps=args.steps)
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save(args.out)


if __name__ == "__main__":
    main()
