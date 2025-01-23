import sympy as sp

x = sp.symbols('x')                 # Define the variable
func = x**2 + 2*x +1                # Define the function

result = sp.diff(func, x)           # Calculate the derivative

print("Derivatives of f(x):")
print(result)
