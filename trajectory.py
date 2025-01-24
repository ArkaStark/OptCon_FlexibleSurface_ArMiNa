import numpy as np
import matplotlib.pyplot as plt
from flexible_dyn import flexible_surface_dynamics_symbolic_filled as dyn

def generate_trajectory(z_eq1, z_eq2, u_eq1, u_eq2, smooth_percentage=0, t_f=10, dt=1e-4):
    """
    Generates a trajectory between two equilibrium points with smooth transitions.

    Args:
        x_eq1 (np.ndarray): Initial state equilibrium point.
        x_eq2 (np.ndarray): Final state equilibrium point.
        u_eq1 (np.ndarray): Initial control input equilibrium.
        u_eq2 (np.ndarray): Final control input equilibrium.
        smooth_percentage (float): Percentage of time allocated to smoothing.
        t_f (float): Final time for the trajectory.
        dt (float): Time step duration.

    Returns:
        tuple: x_reference (np.ndarray), u_reference (np.ndarray).
    """
    
    total_time_steps = int(t_f / dt)
    z_size = z_eq2.shape[0]
    
    # Initialize references
    z_reference = np.zeros((z_size, total_time_steps))
    u_reference = np.zeros((2, total_time_steps))

    # Create the cubic spline for the middle region
    t1 = total_time_steps/2 - total_time_steps*smooth_percentage / (2)
    t2 = total_time_steps/2 + total_time_steps*smooth_percentage / (2)
  
    for i in range(z_size):
        for t in range(total_time_steps):
            if t <= t1:  # Before tf/4
                z_reference[i, t] = z_eq1[i]
                u_reference[:,t] = u_eq1
            elif t > t2:  # After tf-(tf/4)
                z_reference[i, t] = z_eq2[i]
                u_reference[:,t] = u_eq2

    return z_reference, u_reference

def plot_trajectory(z_reference, u_reference, t_f=10, dt=1e-4):
    """
    Plots the generated trajectory and control input.

    Args:
        x_reference (np.ndarray): State trajectory.
        u_reference (np.ndarray): Control input trajectory.
        t_f (float): Final time for the trajectory.
        dt (float): Time step duration.
    """

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))

    names = ['z1', 'z2', 'z3', 'z4', 'u2', 'u4']

    for i in range(z_reference.shape[0]):
        ax = fig.add_subplot(3, 2, i+1)
        ax.plot(time, z_reference[i, :], linewidth=2)
        ax.set_title(names[i])

    for i in range(u_reference.shape[0]):
        ax = fig.add_subplot(3, 2, i+5)
        ax.plot(time, u_reference[i, :], linewidth=2)
        ax.set_title(names[i+4])

    plt.tight_layout()
    plt.show()

def plot_opt_trajectory(x_opt, u_opt, x_ref, u_ref, t_f=10, dt=1e-4):
    """
    Plots the optimized trajectory and control input.

    Args:
        x_opt (np.ndarray): State trajectory.
        u_opt (np.ndarray): Control input trajectory.
        t_f (float): Final time for the trajectory.
        dt (float): Time step duration.
    """

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))

    names = ['z1', 'z2', 'z3', 'z4', 'z1_dot', 'z2_dot', 'z3_dot', 'z4_dot', 'u2', 'u4']

    for i in range(x_opt.shape[0]):
        ax = fig.add_subplot(5, 2, i+1)
        ax.plot(time, x_opt[i, :], linewidth=2)
        ax.plot(time, x_ref[i, :], linewidth=2)
        ax.set_title(names[i])

    for i in range(u_opt.shape[0]):
        ax = fig.add_subplot(5, 2, i+9)
        ax.plot(time, u_opt[i, :], linewidth=2)
        ax.plot(time, u_ref[i, :], linewidth=2)
        ax.set_title(names[i+8])

    plt.tight_layout()
    plt.show()


