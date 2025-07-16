import numpy as np
import random as rd

class Bandit:
    '''
    Bandit class
    Implements the bandit problem.
    
    Attributes:
        k: number of arms
        probs: probabilities of each arm
    '''
    def __init__(self, k):
        '''
        Initialize the bandit.
        Args:
            k: number of arms
        Returns:
            None
        '''
        self.k = k
        self.probs = np.random.rand(self.k)
        
    def pull(self, arm):
        '''
        Pull an arm and return the reward.
        Args:
            arm: the arm to pull
        Returns:
            reward: the reward received
        '''
        return 1 if rd.random() < self.probs[arm] else 0

    def get_probs(self):
        '''
        Get the probabilities of each arm.
        Args:
            None
        Returns:
            probs: the probabilities of each arm
        '''
        return self.probs
    
    def get_k(self):
        '''
        Get the number of arms.
        Args:
            None
        Returns:
            k: the number of arms
        '''
        return self.k
