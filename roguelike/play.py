# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Pygame window so a person can play the game with the keyboard.
# Usage: python -m roguelike.play [--seed N] [--floors N] [--record --name NAME]

import argparse
import os

import numpy as np
import pygame

from .demos import Recorder
from .dungeon import STAIRS, WALL
from .env import RoguelikeEnv
from .game import DOWN, DRINK, LEFT, RIGHT, UP, WAIT
from .seeds import TRAIN_SEEDS

TILE = 48           # pixels per map tile
STATUS_HEIGHT = 76  # pixels for the text under the map

# pixel art lives in assets/ at the top of the repo
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
ASSET_NAMES = ("floor", "wall", "stairs", "player", "rat", "goblin", "potion", "heart")

COLORS = {
    "wall": (40, 40, 48),
    "floor": (200, 196, 184),
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


def load_assets(folder=ASSET_DIR):
    """load the pixel art scaled up to tile size. missing files fall back to plain shapes"""
    images = {}
    for name in ASSET_NAMES:
        path = os.path.join(folder, f"{name}.png")
        if os.path.exists(path):
            image = pygame.image.load(path).convert_alpha()
            images[name] = pygame.transform.scale(image, (TILE, TILE))
            images[name + "_icon"] = image  # small copy for the status line
    images["shade"] = make_shade()
    return images


def make_shade():
    """see-through dark square laid over tiles the player already walked on"""
    shade = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 40))
    return shade


def draw_tile(screen, images, game, row, col, big):
    """draw one map tile using the art if it was loaded"""
    rect = pygame.Rect(col * TILE, row * TILE, TILE, TILE)
    tile = game.grid[row, col]
    if tile == WALL:
        if "wall" in images:
            screen.blit(images["wall"], rect)
        else:
            pygame.draw.rect(screen, COLORS["wall"], rect)
        return

    if "floor" in images:
        screen.blit(images["floor"], rect)
    else:
        pygame.draw.rect(screen, COLORS["floor"], rect)
    if (row, col) in game.visited:
        screen.blit(images["shade"], rect)
    if tile == STAIRS:
        if "stairs" in images:
            screen.blit(images["stairs"], rect)
        else:
            pygame.draw.rect(screen, COLORS["stairs"], rect)
            draw_letter(screen, big, ">", COLORS["background"], rect)


def draw_thing(screen, images, name, pos, fallback, font, letter):
    """draw a sprite at a tile, or a colored circle with a letter if there is no art"""
    rect = pygame.Rect(pos[1] * TILE, pos[0] * TILE, TILE, TILE)
    if name in images:
        screen.blit(images[name], rect)
    else:
        pygame.draw.circle(screen, fallback, rect.center, TILE // 2 - 4)
        draw_letter(screen, font, letter, COLORS["text"], rect)


def draw_status(screen, small, images, game, recording, top):
    """status line with heart and potion icons when the art is loaded"""
    pieces = [f"Floor {game.floor}/{game.num_floors}   ", ("heart", "HP "),
              f"{game.hp}/{game.max_hp}   ", ("potion", "Potions "),
              f"{game.potions}   Turn {game.turns_on_floor}/{game.turns_per_floor}"]
    if recording:
        pieces.append(f"   {recording}")
    x = 8
    for piece in pieces:
        if isinstance(piece, tuple):
            icon, fallback = piece
            if icon + "_icon" in images:
                screen.blit(images[icon + "_icon"], (x, top + 1))
                x += 20
                continue
            piece = fallback
        text = small.render(piece, True, COLORS["text"])
        screen.blit(text, (x, top))
        x += text.get_width()


def draw(screen, fonts, game, message, recording=None, images=None):
    """draw the map, the player, enemies, potions, and the status text"""
    big, small = fonts
    images = images or {"shade": make_shade()}
    screen.fill(COLORS["background"])
    for row in range(game.height):
        for col in range(game.width):
            draw_tile(screen, images, game, row, col, big)

    for pos in game.potion_tiles:
        draw_thing(screen, images, "potion", pos, COLORS["potion"], small, "!")
    for enemy in game.enemies:
        draw_thing(screen, images, enemy.kind.name, enemy.pos, COLORS[enemy.kind.name], big, enemy.kind.symbol)
    draw_thing(screen, images, "player", game.pos, COLORS["player"], big, "@")

    top = game.height * TILE + 6
    draw_status(screen, small, images, game, recording, top)
    for line, text in enumerate((message or "", HELP), start=1):
        screen.blit(small.render(text, True, COLORS["text"]), (8, top + line * 22))


def make_window(game):
    """open a window sized to the map and load the art"""
    pygame.init()
    pygame.display.set_caption("Roguelike")
    screen = pygame.display.set_mode((game.width * TILE, game.height * TILE + STATUS_HEIGHT))
    fonts = (pygame.font.SysFont("consolas", 30, bold=True), pygame.font.SysFont("consolas", 16))
    return screen, fonts, load_assets()


def pick_seed(rng):
    """random training map seed (human play never uses test maps)"""
    return int(rng.choice(TRAIN_SEEDS))


def main():
    """play runs until the window is closed"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="play one specific training map")
    parser.add_argument("--floors", type=int, default=1)
    parser.add_argument("--record", action="store_true", help="save each finished run to demos/")
    parser.add_argument("--name", default="player", help="your name, used in the saved file names")
    args = parser.parse_args()
    if args.seed is not None and args.seed not in TRAIN_SEEDS:
        parser.error("seed must be a training seed (0 to 9999)")

    rng = np.random.default_rng()
    env = RoguelikeEnv(num_floors=args.floors)
    saved = 0  # runs recorded this session

    def new_run(seed):
        """reset the game and start a fresh recording if recording"""
        run = int(rng.integers(1000))
        obs, info = env.reset(seed=seed, options={"run": run})
        recorder = Recorder(args.name, info["map_seed"], run, args.floors) if args.record else None
        return obs, recorder

    obs, recorder = new_run(args.seed if args.seed is not None else pick_seed(rng))
    screen, fonts, images = make_window(env.game)
    message = "New run."
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, recorder = new_run(pick_seed(rng))  # an unfinished run is not saved
                message = "New run."
            elif event.type == pygame.KEYDOWN and event.key in KEYS and not env.game.done:
                action = KEYS[event.key]
                if recorder:
                    recorder.add(obs, action)
                obs, _, _, _, info = env.step(action)
                message = describe(info["events"])
                if recorder and env.game.done:
                    recorder.save(won=env.game.won, died=env.game.died)
                    saved += 1
                    message += f" Saved run {saved}."
        recording = f"REC ({saved} saved)" if args.record else None
        draw(screen, fonts, env.game, message, recording, images)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
