import sympy as sp

x = sp.symbols('x')                         # Define the Variable
func = x**2 + 2*x + 1                       # Define the Function

result = sp.integrate(func, (x, 0, 1))      # Calculate the integral from 0 to 1

print("Integral of f(x) from 0 to 1:")
print(result)
