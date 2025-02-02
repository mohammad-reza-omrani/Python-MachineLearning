import tensorflow as tf

def mse_loss(Q, Q_target):
    """
    Compute Mean Squared Error loss between the predicted Q-values (Q) and target Q-values (Q_target).

    Args:
    - Q: Predicted Q-values (Tensor).
    - Q_target: Target Q-values (Tensor).

    Returns:
    - Loss: A scalar tensor representing the mean squared error.
    """
    # Calculate the mean squared error between Q and Q_target
    return tf.reduce_mean(tf.square(Q - Q_target))

# Example Usage:

# Create example tensors for Q and Q_target (for demonstration purposes)
Q = tf.constant([[0.5, 0.2], [0.1, 0.8]], dtype=tf.float32)  # Predicted Q-values
Q_target = tf.constant([[0.7, 0.3], [0.2, 0.7]], dtype=tf.float32)  # Target Q-values

# Compute the MSE loss
loss = mse_loss(Q, Q_target)

# Print the loss
print("MSE Loss:", loss.numpy())