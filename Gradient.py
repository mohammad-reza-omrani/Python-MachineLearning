import sympy as sp

x, y = sp.symbols('x y')                             # Define the variables
func = x**2 + 2*y**2 + 3*x*y                         # Define the function

gradient = [sp.diff(func, var) for var in (x, y)]    # Calculate the gradient

print("Gradient of f(x, y):")
print(gradient)
