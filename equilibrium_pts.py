import numpy as np
import matplotlib.pyplot as plt
from flexible_dyn import x_next_lambda
from flexible_dyn import grad_xu_lambda

def find_equilibrium_points(z_init, u_init, step_size=0.1, max_iter=20):
    # Initial state and input

    tolerance = 1e-6
    dyn = x_next_lambda()
    grad_x = grad_xu_lambda()[0]
    grad_u = grad_xu_lambda()[1]
    z_eq = z_init
    u_eq = u_init

    x_eq = np.append(z_eq, [0, 0, 0, 0])

    for kk in range(max_iter):

        f = dyn(x_eq, u_init)[4:]
        df = np.array(grad_x(x_eq, u_init)[4:,:4])
            
        try:            
            delta = - step_size * np.linalg.pinv(df) @ f
            x_eq = x_eq + np.append(delta, [0, 0, 0, 0])

        except np.linalg.LinAlgError:
            print("Singular matrix at iteration: ", kk)
            break
        if abs(np.linalg.norm(delta)) < tolerance:
            print(f"Converged after {kk} iterations")
            break

    return np.array(x_eq[:4]), np.array(u_eq)


def plot_eq_pts(z_eqs):
    # Plot the equilibrium points
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.01, 0.01)

    for i in range(len(z_eqs)):
        z_eq = z_eqs[i]
        line, = ax.plot([], [], 'o-', lw=2)
        x_coords = [0, 1, 2, 3, 4, 5]
        y_coords = [0] + list(z_eq) + [0]
        line.set_data(x_coords, y_coords)
        ax.legend([f'z_eq {i+1}' for i in range(len(z_eqs))])
    plt.grid()
    plt.show()


def test():
    z0 = [0, 0, 0, 0]
    z_eq1, u_eq1 = find_equilibrium_points(z0, [-10, -5])
    z_eq2, u_eq2 = find_equilibrium_points(z0, [2, -2])
    z_eq3, u_eq3 = find_equilibrium_points(z0, [3, 6])

    print("Equilibrium point 1: ", z_eq1, "#", u_eq1)
    print("Equilibrium point 2: ", z_eq2, "#", u_eq2)
    print("Equilibrium point 3: ", z_eq3, "#", u_eq3)
    plot_eq_pts([z_eq1, z_eq2, z_eq3])

# test()
