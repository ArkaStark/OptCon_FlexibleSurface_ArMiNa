import numpy as np

# Time step for discretization
dt = 1e-3

def flexible_surface_dynamics(x, u):
    """
    Compute the next state of the flexible surface system using discrete-time dynamics.

    Parameters:
    - x: State vector [z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot].
    - u: Input vector [u2, u4].
    
    Returns:
    - x_next: Next state vector.
    """
    # Parameters for the system
    alpha = 128 * 0.2
    c = 0.1
    m = [0.2, 0.3, 0.2, 0.3]
    d = 0.25
    L_ij = np.array([
        [d, 0, d, 2*d, 3*d, 4*d],
        [2*d, d, 0, d, 2*d, 3*d],
        [3*d, 2*d, d, 0, d, 2*d],
        [4*d, 3*d, 2*d, d, 0, d]
    ])  # Distance matrix

    # Initialize variables
    z = x[:4]          # Positions [z1, z2, z3, z4]
    z_dot = x[4:]      # Velocities [z1_dot, z2_dot, z3_dot, z4_dot]
    F = np.zeros(4)    # Force vector
    F[1] = u[0]        # Actuator input at p2
    F[3] = u[1]        # Actuator input at p4

    # Compute accelerations
    z_ddot = np.zeros(4)
    for i in range(4):
        coupling_force = 0
        for j in range(6):
            if j == 5 or j == 0:
                z[j] = 0 
            if i != j:
                dz = z[i] - z[j]
                denominator = L_ij[i, j] * (L_ij[i, j]**2 - dz**2)
                if abs(L_ij[i, j]**2 - dz**2) < 1e-6:  # Avoid division by small values
                    denominator = 1e-6  # Regularize denominator
                coupling_force += dz / denominator

        # Compute acceleration for point i
        z_ddot[i] = (1 / m[i]) * (F[i] - alpha * coupling_force - c * z_dot[i])

    # Discretize dynamics using Euler integration
    z_next = z + dt * z_dot
    z_dot_next = z_dot + dt * z_ddot

    # Combine into next state vector
    x_next = np.hstack((z_next, z_dot_next))

    return x_next

# Example usage
if __name__ == "__main__":
    # Initial state: [positions, velocities]
    x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    # Inputs for actuators p2 and p4
    u = np.array([1.0, 1.0])  

    # Simulate for 10 time steps
    x_next = x
    print("Initial state:", x_next)
    for kk in range(10):
        x_next = flexible_surface_dynamics(x_next, u)
        print(f"Step {kk + 1}: Next state = {x_next}")
