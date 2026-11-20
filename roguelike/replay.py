# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Watch a run again in the game window or save it as a GIF.
# Used for the mistake analysis (the same map before and after training) and report figures.
# Usage:
#   python -m roguelike.replay --demo demos/FILE.npz
#   python -m roguelike.replay --model checkpoints/ppo_step_100000.zip --seed 1000003 --run 0 --gif out.gif
#   python -m roguelike.replay --agent random --seed 1000003 --run 0

import argparse
import os

import numpy as np

from .env import RoguelikeEnv

END_HOLD = 1.5  # seconds the last frame stays up in a GIF


def outcome(game):
    """how a finished run ended, in words"""
    if game.won:
        return "won"
    if game.died:
        return "died"
    return "ran out of turns"


def run_policy(policy, seed, run, num_floors=1):
    """play one run where policy(obs) picks each action. returns the actions and how it ended"""
    env = RoguelikeEnv(num_floors=num_floors)
    obs, _ = env.reset(seed=seed, options={"run": run})
    actions = []
    while not env.game.done:
        action = int(policy(obs))
        actions.append(action)
        obs, *_ = env.step(action)
    return actions, outcome(env.game)


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
    """policy that asks a saved Stable-Baselines3 PPO model for its best action"""
    from stable_baselines3 import PPO
    model = PPO.load(path)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def random_policy(seed=0):
    """policy that picks random actions"""
    rng = np.random.default_rng(seed)
    return lambda obs: rng.integers(6)


def show_run(seed, run, actions, label, num_floors=1, gif=None, fps=6, window=True):
    """draw every turn of a run in the window, a GIF, or both"""
    if not window:
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # draw without opening a window
    import pygame
    from PIL import Image

    from .play import describe, draw, make_window

    states = replay_states(seed, run, actions, num_floors)
    game, _ = next(states)
    screen, fonts, images = make_window(game)
    clock = pygame.time.Clock()
    frames = []

    def show(game, events):
        """draw one turn and keep it for the GIF"""
        turn_text = describe(events) if events else "Start."
        draw(screen, fonts, game, turn_text, images=images, help_text=f"{label}   Esc quit")
        if gif:
            frames.append(Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB")))
        if window:
            pygame.display.flip()
            clock.tick(fps)
            return not any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                           for e in pygame.event.get())
        return True

    keep_going = show(game, None)
    for game, events in states:
        if not keep_going:
            break
        keep_going = show(game, events)

    if gif:
        frames += [frames[-1]] * int(END_HOLD * fps)
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)

    # leave the last turn on screen until the window is closed
    while window and keep_going:
        keep_going = not any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                             for e in pygame.event.get())
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
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    if args.demo:
        seed, run, floors, actions = load_demo(args.demo)
        label = f"Human  seed {seed}"
    else:
        if args.seed is None:
            parser.error("--seed is needed with --model or --agent")
        seed, run, floors = args.seed, args.run, args.floors
        policy = model_policy(args.model) if args.model else random_policy()
        actions, result = run_policy(policy, seed, run, floors)
        name = os.path.splitext(os.path.basename(args.model))[0] if args.model else args.agent
        label = f"{name}  seed {seed}"
        print(f"{name} on seed {seed} run {run}: {result} in {len(actions)} turns")

    if args.no_window and not args.gif:
        parser.error("--no-window only makes sense with --gif")
    show_run(seed, run, actions, label, floors, gif=args.gif, fps=args.fps, window=not args.no_window)
    if args.gif:
        print(f"saved {args.gif}")


if __name__ == "__main__":
    main()
