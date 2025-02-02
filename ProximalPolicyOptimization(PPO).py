import tensorflow as tf

def ppo_objective(old_probs, new_probs, actions, advantages, epsilon):
    """
    PPO objective function to compute the surrogate loss.
    
    Args:
    - old_probs: The probabilities of the actions taken in the old policy.
    - new_probs: The probabilities of the actions taken in the new policy.
    - actions: The actions taken (used for selecting the correct probability).
    - advantages: The advantage function (estimating how much better the action was).
    - epsilon: The clipping parameter to control how much the policy is allowed to change.
    
    Returns:
    - The negative surrogate loss (since we want to minimize this loss).
    """
    
    # Calculate the probability ratio (new / old)
    ratio = new_probs / old_probs
    
    # Clip the ratio to be within the range [1 - epsilon, 1 + epsilon]
    clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    
    # Calculate the surrogate loss
    surrogate_loss = tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
    
    # Return the negative of the surrogate loss (since we want to minimize this loss)
    return -surrogate_loss