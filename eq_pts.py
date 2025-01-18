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

def equilibrium_equations(z,u):
    r = np.zeros(4)
    dr_dz = np.zeros((4,4))

    for i in range(4):
        coupling_force = 0
        d_coupling_force = 0
        for j in range(4):
            if i != j:
                dz = z[i] - z[j]
                denominator = L_ij[i,j]*(L_ij[i, j]**2 - dz**2)
                print(denominator)
                coupling_force += (dz / denominator)
                d_coupling_force += ((L_ij[i, j]**2 - dz**2)-2*dz**2)/(denominator*(L_ij[i, j]**2 - dz**2))
        ri = u[i] + alpha*coupling_force #confrim once 
        dri_dzi = alpha * d_coupling_force
        dri_dzj = -alpha * d_coupling_force
        r[i] = ri
        for j in range(4):
            if i == j:
                dr_dz[i, i] = dri_dzi          
            if i != j:
                dr_dz[i, j] = dri_dzj

    return (r, dr_dz)

######################################################
# Main code
######################################################

max_iter = 10 #int(5e2)
#stepsize = 0.01
z_eq = np.zeros(4) #np.zeros((4, max_iters))
z0 = np.array([0, 0, 0, 0])
u = np.array([0, 0.2, 0, 0])
z_eq = z0

for kk in range(max_iter):
    r, dr_dz = equilibrium_equations(z_eq,u)
    #direction = np.linalg.inv(dr_dz) @ r
    z_eq = z_eq - np.linalg.inv(dr_dz) @ r
    print("Iteration: ", kk, "z_eq: ", z_eq)

print("Final: ",z_eq)
