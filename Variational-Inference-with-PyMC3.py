import pymc as pm  # For PyMC 4.x or above
import numpy as np
import matplotlib.pyplot as plt
import arviz as az  # Import arviz for handling InferenceData

# Generate synthetic data (replace with actual data if available)
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)  # Generate 100 data points with mean=5 and std=2

# Define the model using a context manager
with pm.Model() as model:
    # Define the prior distribution for the parameter theta (mean of the data)
    theta = pm.Normal('theta', mu=0, sigma=1)

    # Define the likelihood function (assuming normal distribution with unknown mean and known std)
    likelihood = pm.Normal('likelihood', mu=theta, sigma=1, observed=data)

    # Perform variational inference (ADVI method)
    approx = pm.fit(method='advi', n=10000)  # Increase the number of iterations as needed
    trace = approx.sample(1000)  # Sample from the approximate posterior

# Check if trace is a MultiTrace object
print(isinstance(trace, pm.backends.base.MultiTrace))  # Should print True

# Convert MultiTrace to InferenceData using Arviz (alternative method)
trace_inference_data = az.convert_to_inference_data(trace)

# Summary of the trace
az.summary(trace_inference_data).round(2)

# Plot the trace and posterior distribution of theta
az.plot_trace(trace_inference_data)
plt.show()