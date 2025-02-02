import numpy as np

def policy_evaluation(pi, P, R, gamma, tol=1e-6, max_iterations=1000):
    """
    Perform policy evaluation to compute the value function V for a given policy.
    
    Args:
    - pi: Policy (array of action indices for each state).
    - P: Transition probability matrix (shape: [n_states, n_actions, n_states]).
    - R: Reward matrix (shape: [n_states, n_actions, n_states]).
    - gamma: Discount factor.
    - tol: Convergence tolerance (default is 1e-6).
    - max_iterations: Maximum number of iterations (default is 1000).
    
    Returns:
    - V: Value function for the policy.
    """
    n_states = len(P)
    V = np.zeros(n_states)

    for _ in range(max_iterations):
        V_prime = np.copy(V)
        for s in range(n_states):
            a = pi[s]
            # Add a check for valid rewards and probabilities (avoid overflow)
            V[s] = sum(P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V_prime[s_prime]) for s_prime in range(n_states))
        
        # Check for convergence
        if np.max(np.abs(V - V_prime)) < tol:
            break
    
    return V

def policy_iteration(P, R, gamma, tol=1e-6, max_iterations=1000):
    """
    Perform policy iteration to find the optimal policy and value function.
    
    Args:
    - P: Transition probability matrix (shape: [n_states, n_actions, n_states]).
    - R: Reward matrix (shape: [n_states, n_actions, n_states]).
    - gamma: Discount factor.
    - tol: Convergence tolerance (default is 1e-6).
    - max_iterations: Maximum number of iterations (default is 1000).
    
    Returns:
    - pi: Optimal policy.
    - V: Optimal value function.
    """
    n_states, n_actions, _ = P.shape
    pi = np.zeros(n_states, dtype=int)

    for _ in range(max_iterations):
        # Perform policy evaluation
        V = policy_evaluation(pi, P, R, gamma, tol, max_iterations)
        policy_stable = True

        for s in range(n_states):
            a = pi[s]
            # Calculate the current Q-value for the current action
            max_q = np.max([sum(P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime]) for s_prime in range(n_states))])
            
            # Find the best action based on the updated value function
            for a_prime in range(n_actions):
                q = sum(P[s, a_prime, s_prime] * (R[s, a_prime, s_prime] + gamma * V[s_prime]) for s_prime in range(n_states))
                
                if q > max_q:
                    pi[s] = a_prime
                    policy_stable = False
                    break

            if not policy_stable:
                break
        
        # If the policy is stable, stop iterating
        if policy_stable:
            break
    
    return pi, V

def value_iteration(P, R, gamma, tol=1e-6, max_iterations=1000):
    """
    Perform value iteration to find the optimal policy and value function.
    
    Args:
    - P: Transition probability matrix (shape: [n_states, n_actions, n_states]).
    - R: Reward matrix (shape: [n_states, n_actions, n_states]).
    - gamma: Discount factor.
    - tol: Convergence tolerance (default is 1e-6).
    - max_iterations: Maximum number of iterations (default is 1000).
    
    Returns:
    - pi: Optimal policy.
    - V: Optimal value function.
    """
    n_states = len(P)
    V = np.zeros(n_states)

    for _ in range(max_iterations):
        V_prime = np.copy(V)
        for s in range(n_states):
            max_q = np.max([sum(P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V_prime[s_prime]) for s_prime in range(n_states)) for a in range(len(P[0]))])
            V[s] = max_q
        
        # Check for convergence
        if np.max(np.abs(V - V_prime)) < tol:
            break
    
    # Deriving the optimal policy based on the value function
    pi = np.argmax([sum(P[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime]) for s_prime in range(n_states)) for a in range(len(P[0]))], axis=1)

    return pi, V


# Usage Example:

# Example for a small grid world or any environment
n_states = 5  # Number of states
n_actions = 2  # Number of actions
gamma = 0.9  # Discount factor

# Generate random transition probabilities (shape: [n_states, n_actions, n_states])
P = np.random.rand(n_states, n_actions, n_states)

# Normalize the transition probabilities so that each action at each state sums to 1
P /= P.sum(axis=2, keepdims=True)

# Generate random rewards (shape: [n_states, n_actions, n_states])
R = np.random.rand(n_states, n_actions, n_states)

# Run policy iteration to find the optimal policy and value function
pi, V = policy_iteration(P, R, gamma)

print("Optimal Policy:", pi)
print("Optimal Value Function:", V)
