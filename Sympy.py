from sympy import symbols, Matrix, Function, diff

# Define symbolic variables
t = symbols('t')  # Time variable
dt = symbols('dt')  # Time step for discretization

# State variables
z = Matrix([Function(f'z{i+1}')(t) for i in range(4)])  # Positions [z1, z2, z3, z4]
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

# Actuator inputs
u = Matrix([symbols('u2'), symbols('u4')])
F = Matrix([0, u[0], 0, u[1]])  # Force vector

# Coupling forces and accelerations
z_ddot = Matrix.zeros(4, 1)
for i in range(4):
    coupling_force = 0
    for j in range(4):
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

# Display symbolic expressions
print("Symbolic next state vector:")
print(x_next)
