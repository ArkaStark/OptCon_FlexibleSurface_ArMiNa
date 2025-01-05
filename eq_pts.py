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

def equilibrium_equations(z):
    f = np.zeros(4)
    df_dz = np.zeros((4,4))

    for i in range(4):
        coupling_force = 0
        d_coupling_force = 0
        for j in range(4):
            if i != j:
                dz = z[i] - z[j]
                denominator = L_ij[i,j]*(L_ij[i, j]**2 - dz**2)
                coupling_force += (dz / denominator)
                d_coupling_force += (denominator+2*L_ij[i,j]*dz**2)/(denominator**2)
        fi = alpha*coupling_force
        dfi_dzi = alpha * d_coupling_force
        dfi_dzj = -alpha * d_coupling_force
        f[i] = fi
        df_dz[i, i] = dfi_dzi
        for j in range(4):
            if i != j:
                df_dz[i, j] = dfi_dzj

    return (f, df_dz)

max_iter = 10
z_eq = np.zeros(4)
z0 = np.array([1, 0, 0, 0])
z_eq = z0

for kk in range(max_iter):
    f, df_dz = equilibrium_equations(z_eq)
    z_eq = z_eq - np.linalg.inv(df_dz) @ f
    print("Iteration: ", kk, "z_eq: ", z_eq)

print("Final: ",z_eq)