import numpy as np

def rnn_forward(x, W_hx, W_hh, W_yh, b_h, b_y, activation):
    T, d = x.shape  # T: time steps, d: input features
    h = W_hx.shape[0]  # h: hidden state size
    c = W_yh.shape[0]  # c: output size
    
    h_t = np.zeros((T, h))  # Hidden states
    y_hat = np.zeros((T, c))  # Outputs

    for t in range(T):
        if t == 0:
            h_t[t] = activation(np.dot(W_hx, x[t]) + b_h)  # First time step
        else:
            h_t[t] = activation(np.dot(W_hx, x[t]) + np.dot(W_hh, h_t[t-1]) + b_h)  # Subsequent time steps
        
        y_hat[t] = np.dot(W_yh, h_t[t]) + b_y  # Output at each time step

    return h_t, y_hat

# --- Usage Example ---
T = 5  # Time steps
d = 3  # Input features
h = 4  # Hidden state size
c = 2  # Output size

# Random initialization of weights and biases
x = np.random.randn(T, d)
W_hx = np.random.randn(h, d)
W_hh = np.random.randn(h, h)
W_yh = np.random.randn(c, h)
b_h = np.random.randn(h)
b_y = np.random.randn(c)

# Example with np.tanh as the activation function
h_t, y_hat = rnn_forward(x, W_hx, W_hh, W_yh, b_h, b_y, np.tanh)

# Printing the hidden states and outputs
print("Hidden States (h_t):")
print(h_t)

print("\nPredicted Outputs (y_hat):")
print(y_hat)