# PROGRAMMER: Elijah Clare
# DUE DATE: Monday, 11/30/26
# Train a PPO agent on RoguelikeEnv using Stable-Baselines3.

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from roguelike.env import NUM_FLOORS, RoguelikeEnv
from roguelike.reward import ShapedReward, outcome_reward
from roguelike.seeds import TRAIN_SEEDS


class DecayAndEvalCallback(BaseCallback):
    def __init__(self, total_timesteps: int, eval_freq: int = 50_000, prefix: str = "ppo"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.prefix = prefix
        self._next_save = eval_freq

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        self.training_env.env_method("set_reward_progress", progress) if False else None
        if self.num_timesteps >= self._next_save:
            self.model.save(f"checkpoints/{self.prefix}_step_{self.num_timesteps}")
            self._next_save += self.eval_freq
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["sparse", "shaped"], default="shaped")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--floors", type=int, default=NUM_FLOORS)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/ppo_5f_shaped")
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    reward_fn = outcome_reward if args.reward == "sparse" else ShapedReward()

    def make_env():
        return RoguelikeEnv(
            num_floors=args.floors,
            seed_pool=TRAIN_SEEDS,
            reward_fn=reward_fn,
        )

    env = make_vec_env(make_env, n_envs=args.envs, vec_env_cls=SubprocVecEnv)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="logs/")
    callback = DecayAndEvalCallback(
        total_timesteps=args.steps,
        prefix=Path(args.out).name,
    )
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save(args.out)


if __name__ == "__main__":
    main()
