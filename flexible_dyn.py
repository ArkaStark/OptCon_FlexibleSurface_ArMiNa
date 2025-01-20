import numpy as np
import sympy as sp

dt = 1e-4

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

def flexible_surface_dynamics_symbolic():
    # Define symbolic variables
    z = sp.Matrix(sp.symbols('z1:5'))  # z1, z2, z3, z4
    z_dot = sp.Matrix(sp.symbols('dot_z1:5'))  # z1_dot, z2_dot, z3_dot, z4_dot
    m = sp.Matrix(sp.symbols('m1:5'))  # m1, m2, m3, m4
    F = sp.Matrix(sp.symbols('F1:5'))  # F1, F2, F3, F4
    alpha, c, d, dt = sp.symbols('alpha c d dt')


    # Define symbolic adjacency matrix and rest lengths
    L_ij = sp.Matrix([
        [0, d, 2*d, 3*d],
        [d, 0, d, 2*d],
        [2*d, d, 0, d],
        [3*d, 2*d, d, 0]
    ])  # Distance matrix

    # Compute z_ddot symbolically
    z_ddot = sp.Matrix([0, 0, 0, 0])
    for i in range(4):
        coupling_force = 0
        for j in range(4):
            if i == j:
                continue
            dz = z[i] - z[j]
            coupling_force += dz / sp.sqrt(L_ij[i, j]**2 - dz**2)

        z_ddot[i] = (1 / m[i]) * (F[i] - alpha * coupling_force - c * z_dot[i])

    # Compute next state
    z_next = z + dt * z_dot
    z_dot_next = z_dot + dt * z_ddot

    x_next = sp.Matrix.vstack(z_next, z_dot_next)
    return x_next

x_next_sym = flexible_surface_dynamics_symbolic()
# sp.pprint(x_next_sym)

x_next_sym_filled = x_next_sym.subs({
    'm1': 0.2,
    'm2': 0.3,
    'm3': 0.2,
    'm4': 0.3,
    'F1': 0.0,
    'F3': 0.0,
    'alpha': 1.0,
    'c': 0.1,
    'd': 0.25,
    'dt': 1e-4
})




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
u = np.array([100, -100])  # Inputs for p2 and p4
x_next = x0

timesteps=2
x_next_alt = []
# Compute next state
for kk in range(timesteps):
    x_next = flexible_surface_dynamics(x_next, u)

    if kk == 0:
        x_next_alt = x_next_alt + [x_next_sym_filled.subs({
            'z1': x0[0],
            'z2': x0[1],
            'z3': x0[2],
            'z4': x0[3],
            'dot_z1': x0[4],
            'dot_z2': x0[5],
            'dot_z3': x0[6],
            'dot_z4': x0[7],
            'F2': u[0],
            'F4': u[1]
        })]
    else:
        x_next_alt = x_next_alt + [x_next_sym_filled.subs({
            'z1': x_next_alt[kk-1][0],
            'z2': x_next_alt[kk-1][1],
            'z3': x_next_alt[kk-1][2],
            'z4': x_next_alt[kk-1][3],
            'dot_z1': x_next_alt[kk-1][4],
            'dot_z2': x_next_alt[kk-1][5],
            'dot_z3': x_next_alt[kk-1][6],
            'dot_z4': x_next_alt[kk-1][7],
            'F2': u[0],
            'F4': u[1]
        })]

    print("Symbolic dynamics filled:", x_next_alt[kk])
    print("Next state:", x_next)
