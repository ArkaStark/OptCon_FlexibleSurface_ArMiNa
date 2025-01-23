import numpy as np
import matplotlib.pyplot as plt
from flexible_dyn import flexible_surface_dynamics_symbolic_filled as dynamics
from flexible_dyn import dynamics_grad_filled as dynamics_grad

def find_equilibrium_points(u1, u2):
    # Initial state and input
    max_iter = 100
    z0 = np.array([0, 0, 0, 0])
    z_eq = z0


    for kk in range(max_iter):
        x = np.append(z_eq, [0, 0, 0, 0])
        f = dynamics(x, np.array([u1, u2]))[4:]
        df_dz = dynamics_grad(x, np.array([u1, u2]))

        # print(np.linalg.inv(df_dz) @ f)
        if abs(np.sum(np.linalg.inv(df_dz) @ f)) < 1e-8:
            break

        z_eq = z_eq - np.linalg.inv(df_dz) @ f
        print("Iteration: ", kk, "z_eq: ", z_eq)

    return z_eq


def plot_eq_pts(z_eq1, z_eq2):
    # Plot the equilibrium points

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    line1, = ax.plot([], [], 'o-', lw=2)
    line2, = ax.plot([], [], 'o-', lw=2)

    x_coords = [0, 1, 2, 3, 4, 5]
    y1_coords = [0] + list(z_eq1) + [0]
    y2_coords = [0] + list(z_eq2) + [0]
    line1.set_data(x_coords, y1_coords)
    line2.set_data(x_coords, y2_coords)
    plt.grid()
    plt.show()

z_eq1 = find_equilibrium_points(-1000, -500)
z_eq2 = find_equilibrium_points(200, -200)

print("Equilibrium point 1: ", z_eq1)
print("Equilibrium point 2: ", z_eq2)
plot_eq_pts(z_eq1, z_eq2)
