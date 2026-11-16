# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Pygame window so a person can play the game with the keyboard.
# Usage: python -m roguelike.play [--seed N] [--floors N]

import argparse

import numpy as np
import pygame

from .dungeon import STAIRS, WALL
from .env import RoguelikeEnv
from .game import DOWN, DRINK, LEFT, RIGHT, UP, WAIT
from .seeds import TRAIN_SEEDS

TILE = 48           # pixels per map tile
STATUS_HEIGHT = 76  # pixels for the text under the map

COLORS = {
    "wall": (40, 40, 48),
    "floor": (200, 196, 184),
    "visited": (178, 172, 158),
    "stairs": (222, 180, 60),
    "player": (50, 110, 220),
    "rat": (140, 90, 50),
    "goblin": (60, 150, 70),
    "potion": (200, 50, 60),
    "text": (235, 235, 235),
    "background": (20, 20, 24),
}

KEYS = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_SPACE: WAIT,
    pygame.K_q: DRINK,
}

HELP = "Arrows/WASD move  Space wait  Q drink  R new run  Esc quit"


def describe(events):
    """short sentence about what happened this turn"""
    parts = []
    if events["killed_enemy"]:
        parts.append("You killed an enemy.")
    elif events["attacked"]:
        parts.append("You hit an enemy.")
    if events["hit_wall"]:
        parts.append("You bumped a wall.")
    if events["picked_potion"]:
        parts.append("You picked up a potion.")
    if events["drank_potion"]:
        parts.append(f"You healed {events['healed']} HP.")
    if events["damage_taken"]:
        parts.append(f"You took {events['damage_taken']} damage.")
    if events["reached_stairs"] and not events["won"]:
        parts.append("You went down the stairs.")
    if events["won"]:
        parts.append("You cleared the dungeon! Press R for a new run.")
    if events["died"]:
        parts.append("You died. Press R for a new run.")
    if events["timed_out"]:
        parts.append("Out of turns. Press R for a new run.")
    return " ".join(parts)


def draw_letter(screen, font, letter, color, rect):
    """draw a letter centered in a tile"""
    text = font.render(letter, True, color)
    screen.blit(text, text.get_rect(center=rect.center))


def draw(screen, fonts, game, message):
    """draw the map, the player, enemies, potions, and the status text"""
    big, small = fonts
    screen.fill(COLORS["background"])
    for row in range(game.height):
        for col in range(game.width):
            rect = pygame.Rect(col * TILE, row * TILE, TILE, TILE)
            tile = game.grid[row, col]
            if tile == WALL:
                color = COLORS["wall"]
            elif tile == STAIRS:
                color = COLORS["stairs"]
            elif (row, col) in game.visited:
                color = COLORS["visited"]
            else:
                color = COLORS["floor"]
            pygame.draw.rect(screen, color, rect)
            if tile == STAIRS:
                draw_letter(screen, big, ">", COLORS["background"], rect)

    for row, col in game.potion_tiles:
        rect = pygame.Rect(col * TILE, row * TILE, TILE, TILE)
        pygame.draw.circle(screen, COLORS["potion"], rect.center, TILE // 4)
        draw_letter(screen, small, "!", COLORS["text"], rect)

    for enemy in game.enemies:
        rect = pygame.Rect(enemy.pos[1] * TILE, enemy.pos[0] * TILE, TILE, TILE)
        pygame.draw.circle(screen, COLORS[enemy.kind.name], rect.center, TILE // 2 - 4)
        draw_letter(screen, big, enemy.kind.symbol, COLORS["text"], rect)

    rect = pygame.Rect(game.pos[1] * TILE, game.pos[0] * TILE, TILE, TILE)
    pygame.draw.circle(screen, COLORS["player"], rect.center, TILE // 2 - 4)
    draw_letter(screen, big, "@", COLORS["text"], rect)

    top = game.height * TILE + 6
    status = (f"Floor {game.floor}/{game.num_floors}   HP {game.hp}/{game.max_hp}   "
              f"Potions {game.potions}   Turn {game.turns_on_floor}/{game.turns_per_floor}")
    for line, text in enumerate((status, message or "", HELP)):
        screen.blit(small.render(text, True, COLORS["text"]), (8, top + line * 22))


def make_window(game):
    """open a window sized to the map"""
    pygame.init()
    pygame.display.set_caption("Roguelike")
    screen = pygame.display.set_mode((game.width * TILE, game.height * TILE + STATUS_HEIGHT))
    fonts = (pygame.font.SysFont("consolas", 30, bold=True), pygame.font.SysFont("consolas", 16))
    return screen, fonts


def pick_seed(rng):
    """random training map seed (human play never uses test maps)"""
    return int(rng.choice(TRAIN_SEEDS))


def main():
    """play runs until the window is closed"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="play one specific training map")
    parser.add_argument("--floors", type=int, default=1)
    args = parser.parse_args()
    if args.seed is not None and args.seed not in TRAIN_SEEDS:
        parser.error("seed must be a training seed (0 to 9999)")

    rng = np.random.default_rng()
    env = RoguelikeEnv(num_floors=args.floors)
    env.reset(seed=args.seed if args.seed is not None else pick_seed(rng),
              options={"run": int(rng.integers(1000))})
    screen, fonts = make_window(env.game)
    message = "New run."
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset(seed=pick_seed(rng), options={"run": int(rng.integers(1000))})
                message = "New run."
            elif event.type == pygame.KEYDOWN and event.key in KEYS and not env.game.done:
                _, _, _, _, info = env.step(KEYS[event.key])
                message = describe(info["events"])
        draw(screen, fonts, env.game, message)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
