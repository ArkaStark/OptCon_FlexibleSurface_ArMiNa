import numpy as np
from sympy import symbols, Matrix, diff, solve

dt = 1e-3

def flexible_surface_dynamics(x, u):
    """
    Compute the next state of the flexible surface system using discrete-time dynamics.

    Parameters:
    - x: State vector [z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot].
    - u: Input vector [u2, u4].
    - params: Dictionary containing model parameters (alpha, c, m, d, L_ij, neighbors).
    - dt: Time step for discretization.

    Returns:
    - x_next: Next state vector.
    """
    # Parameter set 2
    alpha = 128*0.2
    c = 0.1
    m = [0.2, 0.3, 0.2, 0.3]    # To Check
    d = 0.25
    L_ij = np.array([
        [0, d, 2*d, 3*d],
        [d, 0, d, 2*d],
        [2*d, d, 0, d],
        [3*d, 2*d, d, 0]
    ])  # Distance matrix
    # neighbors = params['neighbors']  # Neighbor sets for each point

    # Initialize variables
    z = x[:4]  # Positions [z1, z2, z3, z4]
    z_dot = x[4:]  # Velocities [z1_dot, z2_dot, z3_dot, z4_dot]

    # Force vector initialization
    F = np.zeros(4)
    F[1] = u[0]  # u2
    F[3] = u[1]  # u4

    # Compute accelerations
    z_ddot = np.zeros(4)
    for i in range(4):
        coupling_force = 0
        for j in range(4):
            if i == j:
                continue
            dz = z[i] - z[j]
            denominator = L_ij[i,j]*(L_ij[i, j]**2 - dz**2)
            coupling_force += (dz / denominator)

        z_ddot[i] = (1 / m[i]) * (F[i] - alpha * coupling_force - c * z_dot[i])

    # Discretize dynamics
    z_next = z + dt * z_dot
    z_dot_next = z_dot + dt * z_ddot

    # Combine into next state vector
    x_next = np.hstack((z_next, z_dot_next))

    return x_next

# Example usage
# params = {
#     'alpha': 1.0,
#     'c': 0.1,
#     'm': [1.0, 1.0, 1.0, 1.0],
#     'd': 1.0,
#     'L_ij': np.array([
#         [0, 1, 0, 0],
#         [1, 0, 1, 0],
#         [0, 1, 0, 1],
#         [0, 0, 1, 0]
#     ]),
#     'neighbors': {
#         0: [1],
#         1: [0, 2],
#         2: [1, 3],
#         3: [2]
#     }
# }

# Initial state and input
x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Initial positions and velocities
u = np.array([1.0, 1.0])  # Inputs for p2 and p4
x_next = x0
# Compute next state
for kk in range(10):
    x_next = flexible_surface_dynamics(x_next, u)
    print("Next state:", x_next)
