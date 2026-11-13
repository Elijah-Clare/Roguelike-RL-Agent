from agents.random_agent import RandomAgent
from evaluation.evaluate import evaluate, summarize

def test_evaluate_runs_and_is_repeatable():
    """
    A test to make sure our evaluation tool works properly:
    1. Checks that it creates the correct number of game runs.
    2. Checks that running it twice gives the exact same results (repeatable/reliable).
    3. Checks that all expected score categories are present.
    """
    # Run the evaluation twice with the same settings to compare them
    first = evaluate(RandomAgent(seed=0), num_seeds=2, runs_per_seed=2)
    second = evaluate(RandomAgent(seed=0), num_seeds=2, runs_per_seed=2)
    
    # Make sure we got 4 total games and that both tests produced identical results
    assert len(first) == 4
    assert first == second
    
    # Make sure all our key score labels are included in the summary
    assert set(summarize(first)) == {"won", "died", "floor", "turns", "new_tiles", "reward"}
