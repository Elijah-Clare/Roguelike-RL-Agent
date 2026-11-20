# Scripted baseline bot: attacks adjacent enemies and navigates to stairs.

from __future__ import annotations
from collections import deque
import numpy as np
from roguelike.env import ENEMY_CH, STAIRS_CH, VIEW_SIZE, VISITED_CH, WALL_CH

# Action indices matching RoguelikeEnv: 0=up, 1=down, 2=left, 3=right, 4=wait, 5=potion
DIRS = [
    (-1, 0, 0),  # up
    (1, 0, 1),   # down
    (0, -1, 2),  # left
    (0, 1, 3),   # right
]
WAIT = 4
DRINK_POTION = 5
CENTER = VIEW_SIZE // 2  # (5, 5)


class ScriptedBot:
    name = "scripted"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def _bfs_first_action(self, walls: np.ndarray, is_goal) -> int | None:
        # Return the first action along a shortest walkable path to any goal tile.
        queue = deque([(CENTER, CENTER, None)])
        seen = {(CENTER, CENTER)}

        while queue:
            r, c, first_action = queue.popleft()
            if (r, c) != (CENTER, CENTER) and is_goal(r, c):
                return first_action

            for dr, dc, action in DIRS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < VIEW_SIZE
                    and 0 <= nc < VIEW_SIZE
                    and (nr, nc) not in seen
                    and walls[nr, nc] == 0.0
                ):
                    seen.add((nr, nc))
                    next_first = action if first_action is None else first_action
                    queue.append((nr, nc, next_first))

        return None

    def act(self, obs: dict) -> int:
        view = obs["view"]
        stats = obs["stats"]
        walls = view[WALL_CH]

        # 1. Drink a potion if HP is at or below 50% and we have one
        if stats[0] <= 0.5 and stats[3] > 0.0:
            return DRINK_POTION

        # 2. Attack any adjacent enemy (bump-to-attack)
        for dr, dc, action in DIRS:
            if view[ENEMY_CH, CENTER + dr, CENTER + dc] > 0.0:
                return action

        # 3. Walk toward stairs if visible in the 11x11 window
        if np.any(view[STAIRS_CH] > 0.0):
            action = self._bfs_first_action(
                walls, lambda r, c: view[STAIRS_CH, r, c] > 0.0
            )
            if action is not None:
                return action

        # 4. Otherwise explore toward the nearest unvisited walkable tile
        action = self._bfs_first_action(
            walls, lambda r, c: view[VISITED_CH, r, c] == 0.0
        )
        if action is not None:
            return action

        # 5. Fallback: pick any open adjacent tile, or wait if boxed in
        open_actions = [
            action
            for dr, dc, action in DIRS
            if walls[CENTER + dr, CENTER + dc] == 0.0
        ]
        if open_actions:
            return int(self.rng.choice(open_actions))
        return WAIT