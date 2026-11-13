import numpy as np

class RandomAgent:
    """
    An agent that plays the game by making completely random choices. 
    It follows the standard 'act and reset' structure that all our game players use.
    """
    name = "random"
    
    def __init__(self, num_actions=6, seed=None):
        self.num_actions = num_actions
        # Set up a random number generator so our random choices can be repeated if we use a seed
        self.rng = np.random.default_rng(seed)
        
    def reset(self):
        """
        Called at the start of a new game. 
        (A random agent doesn't need to remember anything between games, so this is empty).
        """
        pass

    def act(self, obs):
        """
        Looks at the current game screen (obs) and picks a random move.
        """
        return int(self.rng.integers(self.num_actions))
