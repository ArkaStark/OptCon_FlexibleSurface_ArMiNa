import numpy as np
from scipy.optimize import root

# System parameters
params = {
    "alpha": 128 * 0.25,  # Coupling coefficient
    "d": 0.25,             # Distance between points
}
alpha = params["alpha"]
d = params["d"]
L_ij = np.array([
        [0, d, 2*d, 3*d, d, 4*d],
        [d, 0, d, 2*d, 2*d, 3*d],
        [2*d, d, 0, d, 3*d, 2*d],
        [3*d, 2*d, d, 0, 4*d, d]
    ]) 

u = np.array([0.5,-0.5])
# Define the equilibrium equations
def equilibrium_equations(z):
    """
    Residuals for the equilibrium equations.
    Args:
        z (ndarray): Current guess for positions [z1, z2, z3, z4].
    Returns:
        residuals (ndarray): Residuals of the equilibrium equations for z.
    """
    n_points = len(z)
    residuals = np.zeros(n_points)
    z = np.array(list(z)+[0, 0])   # Add the fixed points
    for i in range(n_points):
        interaction_sum = 0
        for j in range(6):
            if j != i:
                numerator = z[i] - z[j]
                denominator = L_ij[i][j]*(L_ij[i][j]**2 - (z[i] - z[j])**2)
                if abs(denominator) < 1e-6:  # Avoid division by zero
                    continue
                interaction_sum += numerator / denominator
        
        # Apply inputs to actuated points (z2, z4)
        if i == 1:  # z2 is actuated by u1
            Fi = u[0]
        elif i == 3:  # z4 is actuated by u2
            Fi = u[1]
        else:
            Fi = 0  # Non-actuated points

        residuals[i] = Fi - alpha * interaction_sum
    
    print(residuals)
    
    return residuals

# Initial guess for equilibrium
z_init = np.array([0.05, 0.2, -0.05, -0.2])  # Modify based on expected equilibrium

# Solve using scipy.optimize.root
solution = root(equilibrium_equations, z_init)

# Output results
if solution.success:
    print("Equilibrium found:", solution.x)
else:
    print("Root finding failed:", solution.message)
