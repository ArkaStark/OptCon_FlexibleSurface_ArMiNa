import numpy as np
import matplotlib.pyplot as plt

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({'font.size': 22})

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


######################################################
# Functions
######################################################

def equilibrium_equations(z):
    """
    Compute the equilibrium residuals for the system.
    Args:
        z (ndarray): Positions of points [z1, z2, z3, z4].
        params (dict): System parameters (m, alpha, d).
    Returns:
        residuals (ndarray): Residuals for each equilibrium equation.
    """

    n_points = len(z)
    residuals = np.zeros_like(z)

    for i in range(n_points):
        interaction_sum = 0
        for j in range(n_points):
            if j != i:
                dz = z[i] - z[j]
                denominator = L_ij[i,j]*(L_ij[i, j]**2 - dz**2)
                if abs(denominator) < 1e-6:  # Avoid division by zero
                    continue
                interaction_sum += dz / denominator

        Fi = 0  # Assume no external force for equilibrium
        residuals[i] = Fi - alpha * interaction_sum

    return residuals


def equilibrium_gradient_and_hessian(z):
    """
    Compute the gradient and Hessian of the equilibrium equations.
    Args:
        z (ndarray): Positions of points [z1, z2, z3, z4].
        params (dict): System parameters (m, alpha, d).
    Returns:
        grad (ndarray): Gradient vector (Jacobian of residuals).
        hess (ndarray): Hessian matrix of residuals.
    """

    n_points = len(z)
    grad = np.zeros_like(z)
    hess = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(n_points):
            if j != i:
                dz = z[i] - z[j]
                denominator = L_ij[i,j]*(L_ij[i, j]**2 - dz**2)
                if abs(denominator) < 1e-6:  # Avoid division by zero
                    continue

                interaction_grad = (denominator + 2*L_ij[i,j]*dz**2) / denominator**2    # To Check

                interaction_hess = (
                    ( denominator**2*(2*L_ij[i,j] * dz+2*L_ij[i,j]) - (denominator + 2*L_ij[i,j]*dz**2)*(4*denominator*L_ij[i,j]*dz)) / denominator**4
                )   # To Check

                grad[i] += alpha * interaction_grad
                hess[i, j] += alpha * interaction_hess
                hess[i, i] += alpha * interaction_hess  # Update diagonal

    print(hess)
    return grad, hess


######################################################
# Newton's Method for Equilibrium
######################################################

max_iters = 100
tol = 1e-9
z_init = np.array([10.0, -10.0, 10.0, -10.0])  # Initial guess for equilibrium

z = z_init.copy()
residual_history = []  # To store residual norms

for kk in range(max_iters):
    # Compute residuals, gradient, and Hessian
    residuals = equilibrium_equations(z)
    grad, hess = equilibrium_gradient_and_hessian(z)
    residual_norm = np.linalg.norm(residuals)
    residual_history.append(residual_norm)

    # Check convergence
    if residual_norm < tol:
        print(f"Converged in {kk} iterations!")
        break

    # Update step using Newton's method
    direction = -np.linalg.inv(hess) @ residuals
    z = z + direction

    # Print iteration details
    print(f"Iter {kk + 1}: Residual norm = {residual_norm:.6e}")

z_star = z
print("\nEquilibrium Computation Results:")
print(f"Equilibrium positions: {z_star}")

######################################################
# Plots
######################################################

# Plot residual norm convergence
plt.figure()
plt.plot(range(len(residual_history)), residual_history, marker='o')
plt.title("Residual Norm Convergence")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.yscale("log")
plt.grid()
plt.show()