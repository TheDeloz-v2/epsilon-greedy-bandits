from bandit import Bandit
from agent import Agent

import matplotlib.pyplot as plt
import numpy as np
import random as rd

def run_experiment(epsilon, steps=1000, k=10, seed=None):
    '''
    Run the experiment.
    Args:
        epsilon: exploration parameter
        steps: number of steps
        k: number of arms
        seed: seed for the random number generator
    Returns:
        bandit.probs: true probabilities of each arm
        agent.Q: estimated values of each arm
        cumulative: cumulative reward over time
        fig: figure with the plots
    '''
    if seed is not None:
        np.random.seed(seed)
        rd.seed(seed)

    bandit = Bandit(k)
    agent = Agent(k, epsilon)

    rewards = np.zeros(steps)
    cumulative = np.zeros(steps)

    for t in range(steps):
        arm = agent.select_arm()
        r = bandit.pull(arm)
        agent.update_estimates(arm, r)
        rewards[t] = r
        cumulative[t] = rewards[:t+1].sum()
    
    # PLot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot : cumulative reward over time
    ax1.plot(cumulative)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'ε = {epsilon} | Final Reward = {int(cumulative[-1])}')
    ax1.grid(True)
    
    # Subplot : estimated vs true probabilities
    x = np.arange(k)
    width = 0.35
    ax2.bar(x - width/2, bandit.probs, width, label='True p')
    ax2.bar(x + width/2, agent.Q, width, label='Estimated Q')
    ax2.set_xlabel('Arm')
    ax2.set_ylabel('Probability / Value')
    ax2.set_title(f'Arm Values: True vs Estimated (ε = {epsilon})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()

    return bandit.probs, agent.Q, cumulative, fig


if __name__ == "__main__":
    
    epsilon = [0, 0.01, 0.1, 0.5]
    steps = 10000
    k = 10
    base_seed = 42
    
    for i, e in enumerate(epsilon):
        seed = base_seed + i
        true_probs, estimated_Q, cumulative_reward, fig = run_experiment(e, steps, k, seed)
        plt.figure(fig.number)

    plt.show()
