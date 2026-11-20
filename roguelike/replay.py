# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Watch a run again in the game window or save it as a GIF.
# For PPO models it also draws what the agent is "thinking": arrows sized by how likely each move is.
# Used for the mistake analysis (the same map before and after training) and report figures.
# Usage:
#   python -m roguelike.replay --demo demos/FILE.npz
#   python -m roguelike.replay --model checkpoints/ppo_step_100000.zip --seed 1000003 --run 0 --gif out.gif
#   python -m roguelike.replay --agent random --seed 1000003 --run 0

import argparse
import os
from itertools import chain

import numpy as np

from .env import RoguelikeEnv
from .game import DRINK, MOVES, NUM_ACTIONS, WAIT

END_HOLD = 1.5               # seconds the last frame stays up in a GIF
MIN_ARROW_PROB = 0.03        # moves less likely than this get no arrow
ARROW_COLOR = (255, 214, 0)
ACTION_NAMES = ("up", "down", "left", "right", "wait", "drink")


def outcome(game):
    """how a finished run ended, in words"""
    if game.won:
        return "won"
    if game.died:
        return "died"
    return "ran out of turns"


def run_policy(policy, seed, run, num_floors=1):
    """play one run where policy(obs) returns (action, probs). returns actions, probs, and how it ended"""
    env = RoguelikeEnv(num_floors=num_floors)
    obs, _ = env.reset(seed=seed, options={"run": run})
    actions, probs = [], []
    while not env.game.done:
        action, action_probs = policy(obs)
        actions.append(int(action))
        probs.append(action_probs)
        obs, *_ = env.step(int(action))
    return actions, probs, outcome(env.game)


def replay_states(seed, run, actions, num_floors=1):
    """the game after each action, starting before the first one. yields (game, events)"""
    env = RoguelikeEnv(num_floors=num_floors)
    env.reset(seed=seed, options={"run": run})
    yield env.game, None
    for action in actions:
        _, _, _, _, info = env.step(action)
        yield env.game, info["events"]


def load_demo(path):
    """seed, run, floor count, and actions from a recorded human run"""
    with np.load(path) as demo:
        return (int(demo["map_seed"]), int(demo["run"]), int(demo["num_floors"]),
                [int(a) for a in demo["actions"]])


def model_policy(path):
    """policy that plays a saved PPO model's most likely action and reports every action's chance"""
    import torch
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")

    def policy(obs):
        """most likely action and the probability of each action"""
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            probs = model.policy.get_distribution(obs_tensor).distribution.probs[0].numpy()
        return int(np.argmax(probs)), probs
    return policy


def random_policy(seed=0):
    """policy that picks random actions (no probabilities to show)"""
    rng = np.random.default_rng(seed)
    return lambda obs: (int(rng.integers(NUM_ACTIONS)), None)


def draw_thinking(screen, font, game, probs):
    """arrows from the player for each move, thicker and brighter when more likely"""
    import pygame

    from .play import TILE

    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    center = pygame.Vector2(game.pos[1] * TILE + TILE / 2, game.pos[0] * TILE + TILE / 2)
    for action, (d_row, d_col) in MOVES.items():
        prob = float(probs[action])
        if prob < MIN_ARROW_PROB:
            continue
        direction = pygame.Vector2(d_col, d_row)
        start = center + direction * TILE * 0.4  # start at the tile edge so the player stays visible
        end = center + direction * TILE * 0.95
        color = (*ARROW_COLOR, int(150 + 105 * prob))
        width = 4 + int(12 * prob)
        side = pygame.Vector2(-d_row, d_col) * (7 + 8 * prob)  # arrow head corners
        tip = end + direction * 10
        for shape_color, grow in (((0, 0, 0, 200), 2), (color, 0)):  # dark outline, then the arrow
            pygame.draw.line(overlay, shape_color, start, end, width + 2 * grow)
            pygame.draw.polygon(overlay, shape_color, [tip + direction * grow, end + side * (1 + grow / 8),
                                                       end - side * (1 + grow / 8)])
        label_pos = center + pygame.Vector2(d_col, d_row) * TILE * 1.45
        text = font.render(f"{prob:.0%}", True, (255, 255, 255))
        box = text.get_rect(center=(int(label_pos.x), int(label_pos.y))).inflate(6, 2)
        pygame.draw.rect(overlay, (0, 0, 0, 170), box, border_radius=4)
        overlay.blit(text, text.get_rect(center=box.center))
    screen.blit(overlay, (0, 0))

    # wait and drink have no direction, so they go in a box in the corner
    lines = [f"wait {float(probs[WAIT]):.0%}", f"drink {float(probs[DRINK]):.0%}"]
    box = pygame.Surface((96, 40), pygame.SRCALPHA)
    box.fill((0, 0, 0, 170))
    for i, line in enumerate(lines):
        box.blit(font.render(line, True, (255, 255, 255)), (6, 3 + i * 18))
    screen.blit(box, (4, 4))


class TurnDrawer:
    """draws turns of a run onto one pygame screen (hidden when there is no window)"""

    def __init__(self, game, window):
        """open the screen and load the art"""
        if window:
            os.environ.pop("SDL_VIDEODRIVER", None)  # undo a hidden screen from an earlier GIF
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # draw without opening a window
        from .play import make_window
        self.screen, self.fonts, self.images = make_window(game)

    def draw(self, game, events, label, probs=None):
        """draw one turn, with the thinking arrows when probs are given"""
        from .play import describe, draw
        turn_text = describe(events).replace(" Press R for a new run.", "") if events else "Start."
        draw(self.screen, self.fonts, game, turn_text, images=self.images, help_text=label)
        if probs is not None and not game.done:
            draw_thinking(self.screen, self.fonts[1], game, probs)

    def image(self):
        """the screen as a PIL image"""
        import pygame
        from PIL import Image
        return Image.frombytes("RGB", self.screen.get_size(), pygame.image.tobytes(self.screen, "RGB"))


def run_frames(seed, run, actions, label, probs=None, num_floors=1):
    """every turn of a run as PIL images, without opening a window"""
    states = replay_states(seed, run, actions, num_floors)
    game, events = next(states)
    drawer = TurnDrawer(game, window=False)
    frames = []
    # draw each turn as it happens, since the game object changes after every step
    for turn, (game, events) in enumerate(chain([(game, events)], states)):
        drawer.draw(game, events, label, probs[turn] if probs and turn < len(probs) else None)
        frames.append(drawer.image())
    import pygame
    pygame.quit()
    return frames


def save_gif(frames, path, fps):
    """write frames as a looping GIF that holds on the last frame"""
    from PIL import Image
    frames = frames + [frames[-1]] * int(END_HOLD * fps)
    # fewer colors and no dithering keeps pixel art GIFs small
    frames = [frame.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
              for frame in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                   optimize=True)


def show_run(seed, run, actions, label, num_floors=1, probs=None, fps=6):
    """play a run back in the game window, turn by turn"""
    import pygame

    def quit_pressed():
        """true if the window was closed or Esc was pressed"""
        return any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                   for e in pygame.event.get())

    states = replay_states(seed, run, actions, num_floors)
    game, events = next(states)
    drawer = TurnDrawer(game, window=True)
    clock = pygame.time.Clock()
    for turn, (game, events) in enumerate(chain([(game, events)], states)):
        drawer.draw(game, events, f"{label}   Esc quit", probs[turn] if probs and turn < len(probs) else None)
        pygame.display.flip()
        clock.tick(fps)
        if quit_pressed():
            pygame.quit()
            return

    # leave the last turn on screen until the window is closed
    while not quit_pressed():
        clock.tick(30)
    pygame.quit()


def main():
    """pick a run from the command line and show it"""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", help="recorded human run (.npz)")
    source.add_argument("--model", help="saved PPO model (.zip)")
    source.add_argument("--agent", choices=["random"], help="built in agent")
    parser.add_argument("--seed", type=int, help="map seed (needed with --model or --agent)")
    parser.add_argument("--run", type=int, default=0, help="enemy moves for this run")
    parser.add_argument("--floors", type=int, default=1)
    parser.add_argument("--gif", help="save the run as a GIF here")
    parser.add_argument("--no-window", action="store_true", help="only save the GIF")
    parser.add_argument("--no-thinking", action="store_true", help="hide the PPO move arrows")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    if args.no_window and not args.gif:
        parser.error("--no-window only makes sense with --gif")

    probs = None
    if args.demo:
        seed, run, floors, actions = load_demo(args.demo)
        label = f"Human  seed {seed}"
    else:
        if args.seed is None:
            parser.error("--seed is needed with --model or --agent")
        seed, run, floors = args.seed, args.run, args.floors
        policy = model_policy(args.model) if args.model else random_policy()
        actions, probs, result = run_policy(policy, seed, run, floors)
        name = os.path.splitext(os.path.basename(args.model))[0] if args.model else args.agent
        label = f"{name}  seed {seed}"
        print(f"{name} on seed {seed} run {run}: {result} in {len(actions)} turns")
        if args.no_thinking or probs[0] is None:
            probs = None

    if args.gif:
        save_gif(run_frames(seed, run, actions, label, probs, floors), args.gif, args.fps)
        print(f"saved {args.gif}")
    if not args.no_window:
        show_run(seed, run, actions, label, floors, probs, args.fps)


if __name__ == "__main__":
    main()
