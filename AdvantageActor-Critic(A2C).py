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


class ValueFunction:
    def __init__(self, n_states):
        # Initialize value function weights randomly
        self.weights = np.random.randn(n_states)
    
    def predict(self, state):
        """
        Predict the value of a state.
        
        Args:
        - state: The current state.
        
        Returns:
        - Predicted value of the state.
        """
        return self.weights[state]

def a2c_update(policy, value_function, alpha_p, alpha_v, states, actions, rewards, returns):
    """
    Perform an A2C update for both policy and value function.

    Args:
    - policy: The policy object.
    - value_function: The value function object.
    - alpha_p: Learning rate for the policy.
    - alpha_v: Learning rate for the value function.
    - states: List of states visited.
    - actions: List of actions taken.
    - rewards: List of rewards received.
    - returns: List of return values.
    
    Returns:
    - None (Policy and value function are updated in-place).
    """
    # Update policy
    gradient_p = np.zeros_like(policy.weights)
    for t in range(len(states)):
        state = states[t]
        action = actions[t]
        
        # Compute log probability of the taken action
        log_prob = policy.log_prob(state, action)
        
        # Compute policy gradient
        gradient_p += log_prob * (rewards[t] - value_function.predict(state))
    
    # Average the gradient over the time steps
    gradient_p /= len(states)
    
    # Update policy weights
    policy.weights += alpha_p * gradient_p

    # Update value function
    gradient_v = np.zeros_like(value_function.weights)
    for t in range(len(states)):
        state = states[t]
        target = rewards[t] + returns[t]
        
        # Compute the value function gradient
        gradient_v += (value_function.predict(state) - target) * state
    
    # Average the gradient over the time steps
    gradient_v /= len(states)
    
    # Update value function weights
    value_function.weights += alpha_v * gradient_v

# Example Usage:

# Define environment parameters
n_states = 5
n_actions = 3

# Create policy and value function objects
policy = Policy(n_states, n_actions)
value_function = ValueFunction(n_states)

# Define example trajectory (states, actions, rewards, returns)
states = [0, 1, 2, 3]
actions = [0, 2, 1, 0]
rewards = [1, 0, 0, 1]
returns = [1.2, 0.5, 0.7, 1.0]  # Example returns (sum of future rewards)

# Set learning rates
alpha_p = 0.01  # Learning rate for policy
alpha_v = 0.01  # Learning rate for value function

# Perform the A2C update
a2c_update(policy, value_function, alpha_p, alpha_v, states, actions, rewards, returns)

# Output the updated policy and value function weights
print("Updated Policy Weights:")
print(policy.weights)
print("Updated Value Function Weights:")
print(value_function.weights)