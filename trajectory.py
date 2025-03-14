import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from flexible_dyn import flexible_surface_dynamics_symbolic_filled as dyn

def generate_step_trajectory(z_eqs, u_eqs, t_f=10, dt=1e-4):
    
    TT = int(t_f / dt)
    eq_size = len(z_eqs)
    z_size = z_eqs[0].shape[0]
    u_size = u_eqs[0].shape[0]
    
    # Initialize references
    z_reference = np.zeros((z_size, TT))
    u_reference = np.zeros((u_size, TT))

    t_eq = TT//eq_size
  
    for i in range(eq_size):
        z_eq = z_eqs[i]
        u_eq = u_eqs[i]

        for j in range(z_size):
            z_reference[j, i*t_eq:(i+1)*t_eq] = z_eq[j]
        for j in range(u_size):
            u_reference[j, i*t_eq:(i+1)*t_eq] = u_eq[j]

    return z_reference, u_reference

def generate_smooth_trajectory(z_eqs, u_eqs, t_f=10, dt=1e-4):
 
    eq_size = len(z_eqs)
    TT = int(t_f / dt)
    z_size = z_eqs[0].size
    u_size = u_eqs[0].size
    t_eq = TT//eq_size
    
    # Initialize references
    z_reference = np.zeros((z_size, TT))
    u_reference = np.zeros((u_size, TT))

    for t in range (TT):
        z_reference[:, t] = z_eqs[0]
        u_reference[:, t] = u_eqs[0]

    # Elaborate the reference to match the equilibria transition
    center = [0]
    for i in range(1,  eq_size):
        center.append(i * TT//eq_size)
    for k in range(1, len(center)):
        t1 = int(center[k] - t_eq/ 2)
        t2 = int(center[k] + t_eq/ 2)
      
        for i in range(z_size):
            if t_eq != 0:
                spline = CubicSpline([t1, t2], np.vstack([z_eqs[k-1], z_eqs[k]]), bc_type='clamped')
            for t in range(t1, t2):
                z_reference[:, t] = spline(t)
            for t in range(t2, TT):
                z_reference[:, t] = z_reference[:, t-1]  

        for i in range(u_size):
            if t_eq != 0:
                spline = CubicSpline([t1, t2], np.vstack([u_eqs[k-1], u_eqs[k]]), bc_type='clamped')
            for t in range(t1, t2):
                u_reference[:, t] = spline(t)
            for t in range(t2, TT):
                u_reference[:, t] = u_reference[:, t-1]

    return z_reference, u_reference

def plot_trajectory(z_refs, u_refs, t_f=10, dt=1e-4):

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))

    names = ['z1', 'z2', 'z3', 'z4', 'u2', 'u4']

    for i in range(z_refs[0].shape[0]):
        ax = fig.add_subplot(3, 2, i+1)
        for j in range(len(z_refs)):
            ax.plot(time, z_refs[j][i, :])
        ax.set_title(names[i])

    for i in range(u_refs[0].shape[0]):
        ax = fig.add_subplot(3, 2, i+5)
        for j in range(len(u_refs)):
            ax.plot(time, u_refs[j][i, :])
        ax.set_title(names[i+4])

    plt.tight_layout()
    plt.show()

def plot_opt_trajectory(x_opt, u_opt, x_ref, u_ref, t_f=10, dt=1e-4):

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))

    names = ['z1', 'z2', 'z3', 'z4', 'z1_dot', 'z2_dot', 'z3_dot', 'z4_dot', 'u2', 'u4']

    for i in range(x_opt.shape[0]):
        ax = fig.add_subplot(5, 2, i+1)
        ax.plot(time, x_opt[i, :], linewidth=2)
        ax.plot(time, x_ref[i, :], linewidth=2, linestyle='--')
        ax.set_title(names[i])

    for i in range(u_opt.shape[0]):
        ax = fig.add_subplot(5, 2, i+9)
        ax.plot(time, u_opt[i, :], linewidth=2)
        ax.plot(time, u_ref[i, :], linewidth=2, linestyle='--')
        ax.set_title(names[i+8])

    plt.tight_layout()
    plt.show()


def plot_tracking_error(x_opt, u_opt, x_ref, u_ref, t_f=10, dt=1e-4):

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))

    names = ['z1', 'z2', 'z3', 'z4', 'z1_dot', 'z2_dot', 'z3_dot', 'z4_dot', 'u2', 'u4']

    for i in range(x_opt.shape[0]):
        ax = fig.add_subplot(5, 2, i+1)
        x_error = x_opt[i, :] - x_ref[i, :]
        ax.plot(time, x_error, linewidth=2)
        ax.set_title(names[i])

    for i in range(u_opt.shape[0]):
        ax = fig.add_subplot(5, 2, i+9)
        u_error = u_opt[i, :] - u_ref[i, :]
        ax.plot(time, u_error, linewidth=2)
        ax.set_title(names[i+8])

    plt.tight_layout()
    plt.show()        


