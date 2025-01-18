from sympy import symbols, Matrix, Function, diff
import numpy as np

# Define symbolic variables
t = symbols('t')  # Time variable
dt = symbols('dt')  # Time step for discretization


# State variables
z = Matrix([0, 0, 0, 0, 0, 0, 0, 0])  # Positions [z0, z1, z2, z3, z4, z5]
#z[0], z[5] = 0
z_dot = Matrix([diff(z[i], t) for i in range(4)])       # Velocities [z1_dot, z2_dot, z3_dot, z4_dot]

# Parameters
alpha = symbols('alpha')  # Mechanical coupling coefficient
c = symbols('c')          # Damping coefficient
m = Matrix([symbols(f'm{i+1}') for i in range(4)])  # Masses [m1, m2, m3, m4]
d = symbols('d')  # Distance between direct neighbors
Lij = Matrix([
    [d, 0, d, 2*d, 3*d, 4*d],
        [2*d, d, 0, d, 2*d, 3*d],
        [3*d, 2*d, d, 0, d, 2*d],
        [4*d, 3*d, 2*d, d, 0, d]
])  # Distance matrix
Lij.subs(d, 0.25)
# Actuator inputs
u = Matrix([symbols('u2'), symbols('u4')])
F = Matrix([0, 0.1, 0, -0.1])  # Force vector
#F.subs(u[0]=0.1, u[1]=-0.1)

def flexible_surface_dynamics(z, F):

    
   
    # Coupling forces and accelerations
    z_ddot = Matrix.zeros(4, 1)
    for i in range(4):
        coupling_force = 0
        for j in range(6):
            if i != j:
                dz = z[i] - z[j]
                denominator = Lij[i, j] * (Lij[i, j]**2 - dz**2)
                coupling_force += dz / denominator
                

        # Compute acceleration for point i
        z_ddot[i] = (1 / m[i]) * (F[i] - alpha * coupling_force - c * z_dot[i])

    # Discretize dynamics using Euler integration
    z_next = z + dt * z_dot
    
    z_dot_next = z_dot + dt * z_ddot

    # Combine into the next state vector
    x_next = Matrix.vstack(z_next, z_dot_next)
    return x_next

# Example usage
if __name__ == "__main__":
    
   
    # Simulate for 10 time steps
    x_next = flexible_surface_dynamics(z,F)
    print("Initial state:", x_next)
    for kk in range(10):
        x_next = flexible_surface_dynamics(x_next, u)
        print(f"Step {kk + 1}: Next state = {x_next}")


