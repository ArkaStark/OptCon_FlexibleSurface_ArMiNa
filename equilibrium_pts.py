import numpy as np
import matplotlib.pyplot as plt
from flexible_dyn import flexible_surface_dynamics_symbolic_filled as dynamics
from flexible_dyn import dynamics_grad_filled as dynamics_grad

def find_equilibrium_points(z_init, u, step_size=1, max_iter=100):
    # Initial state and input

    z_eq = z_init
    tolerance = 1e-6

    for kk in range(max_iter):
        x = np.append(z_eq, [0, 0, 0, 0])
        f = dynamics(x, u)[4:]
        df_dz = dynamics_grad(x, u)

        try:
            delta_z = - step_size * np.linalg.pinv(df_dz) @ f
            z_eq = z_eq + delta_z
        except np.linalg.LinAlgError:
            print("Singular matrix at iteration: ", kk)
            break
        if abs(np.sum(delta_z)) < tolerance:
            print(f"Converged after {kk} iterations")
            break

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


def test():
    z0 = [0, 0, 0, 0]
    z_eq1 = find_equilibrium_points(z0, [-1000, -500])
    z_eq2 = find_equilibrium_points(z0, [200, -200])

    print("Equilibrium point 1: ", z_eq1)
    print("Equilibrium point 2: ", z_eq2)
    plot_eq_pts(z_eq1, z_eq2)

# test()
