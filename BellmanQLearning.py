import numpy as np

def q_learning(Q, s, a, r, s_prime, alpha, gamma):
    """
    Performs one step of Q-learning and updates the Q-table.
    
    Args:
    - Q: The Q-table (a 2D numpy array of state-action values).
    - s: Current state.
    - a: Action taken at the current state.
    - r: Reward received after taking action a in state s.
    - s_prime: Next state after taking action a.
    - alpha: Learning rate (how much new information overrides old).
    - gamma: Discount factor (how much future rewards are valued).
    
    Returns:
    - None (Q-table is updated in place).
    """
    # Find the maximum Q-value for the next state
    max_q_prime = np.max(Q[s_prime])
    
    # Update the Q-table using the Q-learning formula
    Q[s, a] += alpha * (r + gamma * max_q_prime - Q[s, a])

# Example Usage:

# Define the size of the Q-table
n_states = 5
n_actions = 3

# Initialize the Q-table with zeros
Q = np.zeros((n_states, n_actions))

# Define some parameters for learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Example of one step of Q-learning
s = 0  # Current state
a = 1  # Action taken
r = 10  # Reward received
s_prime = 2  # Next state after action is taken

# Perform Q-learning update
q_learning(Q, s, a, r, s_prime, alpha, gamma)

# Output the updated Q-table
print("Updated Q-table:")
print(Q)