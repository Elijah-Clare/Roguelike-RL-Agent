# Evaluate an agent on the held out test seeds.
# Usage: python -m evaluation.evaluate --agent random --seeds 100 --runs 5

import argparse
import csv
import os
import numpy as np

from agents.random_agent import RandomAgent
from agents.scripted_bot import ScriptedBot
from agents.ppo_agent import PPOAgent
from roguelike.env import RoguelikeEnv
from roguelike.seeds import TEST_SEED_START

def evaluate(agent, num_seeds=100, runs_per_seed=5, reward_fn=None):
    """
    Runs a series of test games to see how well the agent performs. 
    It plays multiple games across different starting seeds and tracks the results.
    """
    kwargs = {"reward_fn": reward_fn} if reward_fn else {}
    env = RoguelikeEnv(**kwargs)
    results = []
    
    # Go through each test seed to make sure we test on a variety of maps/layouts
    for seed in range(TEST_SEED_START, TEST_SEED_START + num_seeds):
        for run in range(runs_per_seed):
            # Reset the game world and the agent for a fresh start
            obs, _ = env.reset(seed=seed, options={"run": run})
            agent.reset()
            
            total_reward, new_tiles, done = 0.0, 0, False
            
            # Keep playing this specific game until it ends (win, lose, or timeout)
            while not done:
                obs, reward, terminated, truncated, info = env.step(agent.act(obs))
                total_reward += reward
                new_tiles += int(info["events"]["new_tile"])
                done = terminated or truncated
                
            # Save all the important stats from this finished run
            results.append({"seed": seed, "run": run, "won": int(info["won"]),
                            "died": int(info["events"]["died"]), "floor": info["floor"],
                            "turns": info["turns"], "new_tiles": new_tiles,
                            "reward": total_reward})
    return results

def summarize(results):
    """
    Takes all individual game results and calculates the overall averages 
    so we can see how the agent did on average.
    """
    keys = ["won", "died", "floor", "turns", "new_tiles", "reward"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}

def save_csv(results, path):
    """
    Saves all the game data into a spreadsheet file (CSV format) 
    so we can look at it later. Creates the folder if it doesn't exist yet.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
def make_agent(name: str, seed: int = 0, model_path: str | None = None):
    """
    Creates and returns the right type of agent based on a name (like 'random'),
    giving it a starting random seed if needed.
    """
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "scripted":
        return ScriptedBot(seed=seed)
    if name == "ppo":
        if not model_path:
            raise ValueError("--model <path> is required when --agent ppo")
        return PPOAgent(model_path)
    raise ValueError(f"Unknown agent: {name}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="random")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="results")
    parser.add_argument("--model", default=None, help="Path to .zip model for ppo agent")
    args = parser.parse_args()

    # Run the evaluation games and save the data to a file
    agent = make_agent(args.agent, seed=0, model_path=args.model)
    results = evaluate(agent, args.seeds, args.runs)
    save_csv(results, f"{args.out}/{args.agent}_runs.csv")
    
    # Calculate the averages and print them out to the screen
    summary = summarize(results)
    print(f"{args.agent}: {len(results)} runs")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")
