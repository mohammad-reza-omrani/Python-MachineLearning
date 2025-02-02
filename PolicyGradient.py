import numpy as np

class Policy:
    def __init__(self, n_states, n_actions):
        # Initialize policy weights randomly
        self.weights = np.random.randn(n_states, n_actions)
    
    def log_prob(self, state, action):
        """
        Compute the log probability of taking action in a given state.
        
        Args:
        - state: Current state.
        - action: Action taken in that state.
        
        Returns:
        - Log probability of the action.
        """
        # Compute the softmax probabilities
        exp_weights = np.exp(self.weights[state])
        softmax_probs = exp_weights / np.sum(exp_weights)
        
        # Return the log probability of the selected action
        return np.log(softmax_probs[action])

def reinforce_update(policy, alpha, states, actions, rewards):
    """
    Perform a REINFORCE update to update the policy based on the rewards.
    
    Args:
    - policy: The current policy object.
    - alpha: Learning rate.
    - states: List of states visited.
    - actions: List of actions taken.
    - rewards: List of rewards received.
    
    Returns:
    - None (Policy is updated in-place).
    """
    gradient = np.zeros_like(policy.weights)
    
    for t in range(len(states)):
        state = states[t]
        action = actions[t]
        reward = rewards[t]
        
        # Compute the log probability of the taken action
        log_prob = policy.log_prob(state, action)
        
        # Update the gradient
        gradient += log_prob * reward
    
    # Average the gradient over the number of time steps
    gradient /= len(states)
    
    # Update the policy weights using the gradient
    policy.weights += alpha * gradient

# Example Usage:

# Define environment parameters
n_states = 5
n_actions = 3

# Create a policy object
policy = Policy(n_states, n_actions)

# Define example trajectory (states, actions, rewards)
states = [0, 1, 2, 3]
actions = [0, 2, 1, 0]
rewards = [1, 0, 0, 1]

# Set learning rate
alpha = 0.01

# Perform the REINFORCE update
reinforce_update(policy, alpha, states, actions, rewards)

# Output the updated policy weights
print("Updated Policy Weights:")
print(policy.weights)
