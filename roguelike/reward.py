# PROGRAMMER: Elijah Clare
# DUE DATE: Monday, 11/30/26
# Reward functions for RoguelikeEnv (outcome-only and shaped with decay).

from __future__ import annotations

def outcome_reward(events: dict) -> float:
    # Sparse outcome-only reward: +10 for reaching stairs, -5 for dying.
    reward = 0.0
    if events["reached_stairs"]:
        reward += 10.0
    if events["died"]:
        reward -= 5.0
    return reward


class ShapedReward:
    # Full shaped reward from the project design with linear exploration decay.
    # Calling an instance matches RoguelikeEnv's `reward_fn(events)` signature.
    def __init__(
        self,
        explore_start: float = 0.05,
        explore_end: float = 0.0,
    ) -> None:
        self.explore_start = explore_start
        self.explore_end = explore_end
        self.explore_weight = explore_start

    def set_progress(self, progress: float) -> None:
        #Update exploration bonus linearly for training progress in [0.0, 1.0]
        clamped = max(0.0, min(1.0, float(progress)))
        self.explore_weight = (
            self.explore_start
            + (self.explore_end - self.explore_start) * clamped
        )

    def __call__(self, events: dict) -> float:
        reward = outcome_reward(events)

        if events["killed_enemy"]:
            reward += 1.0
        if events["new_tile"]:
            reward += self.explore_weight
        if events["damage_taken"]:
            reward -= 0.1 * events["damage_taken"]
        # Encourages faster completion each step
        reward -= 0.01
        return reward


def shaped_reward(events: dict) -> float:
    # Stateless shaped reward with fixed +0.05 exploration bonus.
    return ShapedReward()(events)
