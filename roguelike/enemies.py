# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Enemy types and how enemies act on their turn.
# Enemies attack when next to the player, chase when close, and wander otherwise.

from dataclasses import dataclass

from .dungeon import WALL

RANDOM_MOVE_CHANCE = 0.2  # chance a chasing enemy moves randomly instead
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class EnemyType:
    """stats shared by every enemy of one kind"""
    name: str
    symbol: str          # letter used in the text map
    max_hp: int
    damage: int          # hp the player loses per hit
    sight: int           # chases the player when this many steps away or closer
    obs_value: float     # value shown in the agent's enemy channel


RAT = EnemyType("rat", "r", max_hp=3, damage=1, sight=4, obs_value=0.5)
GOBLIN = EnemyType("goblin", "g", max_hp=6, damage=2, sight=6, obs_value=1.0)
ENEMY_TYPES = (RAT, GOBLIN)


@dataclass
class Enemy:
    """one enemy on the current floor"""
    kind: EnemyType
    pos: tuple[int, int]
    hp: int


def distance(a, b):
    """steps between two tiles ignoring walls"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def enemy_turn(enemy, game, rng):
    """attack if next to the player, otherwise move. returns the damage dealt"""
    if distance(enemy.pos, game.pos) == 1:
        return enemy.kind.damage

    if distance(enemy.pos, game.pos) <= enemy.kind.sight and rng.random() >= RANDOM_MOVE_CHANCE:
        for step in chase_steps(enemy.pos, game.pos):
            if try_move(enemy, step, game):
                break
    else:
        try_move(enemy, STEPS[int(rng.integers(len(STEPS)))], game)
    return 0


def chase_steps(start, target):
    """steps that get closer to the target, the bigger gap first"""
    d_row, d_col = target[0] - start[0], target[1] - start[1]
    row_step = ((1 if d_row > 0 else -1), 0) if d_row else None
    col_step = (0, (1 if d_col > 0 else -1)) if d_col else None
    steps = [row_step, col_step] if abs(d_row) >= abs(d_col) else [col_step, row_step]
    return [step for step in steps if step]


def try_move(enemy, step, game):
    """move the enemy one step if the tile is open. returns true if it moved"""
    target = (enemy.pos[0] + step[0], enemy.pos[1] + step[1])
    if game.grid[target] == WALL or target == game.pos or game.enemy_at(target):
        return False
    enemy.pos = target
    return True
