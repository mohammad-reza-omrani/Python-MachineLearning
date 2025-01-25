import sympy as sp

x, y = sp.symbols('x y')                 # Define the variables
func = x**2 + 2*y**2 + 3*x*y             # Define the function

hessian = sp.hessian(func, (x, y))       # Calculate the Hessian matrix

print("Hessian Matrix of f(x, y):")
print(hessian)
