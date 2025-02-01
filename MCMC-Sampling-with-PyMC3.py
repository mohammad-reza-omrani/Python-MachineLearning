import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az  # Import arviz for handling InferenceData

# Generate some synthetic data
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)  # Generate 100 data points with mean=5 and std=2

# Define the model using a context manager
with pm.Model() as model:
    
    # Define the prior distribution for the parameter theta (mean of the data)
    theta = pm.Normal('theta', mu=0, sigma=1)
    
    # Define the likelihood function (assuming normal distribution with unknown mean and known std)
    likelihood = pm.Normal('likelihood', mu=theta, sigma=1, observed=data)
    
    # Perform MCMC sampling to estimate theta and return InferenceData directly
    trace_inference_data = pm.sample(1000, tune=1000, return_inferencedata=True)

# Summary of the trace (directly from InferenceData)
az.summary(trace_inference_data).round(2)

# Plot the trace and posterior distribution of theta
az.plot_trace(trace_inference_data)
plt.show()