#Stable-Baselines3 PPO agent wrapper for RoguelikeEnv evaluation.

from __future__ import annotations
from pathlib import Path
from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, model_path: str, deterministic: bool = True) -> None:
        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.name = Path(model_path).stem

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)