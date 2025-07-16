import numpy as np
import random as rd

class Agent:
    '''
    Agent class
    Implements the strategy epsilon-greedy.
    
    Attributes:
        k: number of arms
        epsilon: exploration parameter
        Q: estimated values of each arm
        N: number of times each arm has been selected
    '''
    def __init__(self, k, epsilon):
        '''
        Initialize the agent.
        Args:
            k: number of arms
            epsilon: exploration parameter
        Returns:
            None
        '''
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select_arm(self):
        '''
        Select an arm to pull.
        Args:
            None
        Returns:
            arm: the arm to pull
        '''
        if rd.random() < self.epsilon:
            return rd.randrange(self.k)
        else:
            return int(np.argmax(self.Q))

    def update_estimates(self, arm, reward):
        '''
        Update the estimates of the arm values.
        Args:
            arm: the arm to update
            reward: the reward received
        Returns:
            None
        '''
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]
