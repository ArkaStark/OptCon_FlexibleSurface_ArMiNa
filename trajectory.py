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

def plot_LQR_trajectories(x_real_LQR, u_real_LQR, x_gen, u_gen, t_f=10, dt=1e-4):

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))
    
    # Define naming and color schemes
    names = {
        0: r'$\dot{z}_1$', 1: r'$\dot{z}_2$', 2: r'$\dot{z}_3$', 3: r'$\dot{z}_4$',
        4: r'$z_1$', 5: r'$z_2$', 6: r'$z_3$', 7: r'$z_4$',
        8: r'$u_1$', 9: r'$u_2$'
    }
    colors_ref = {0: 'm', 1: 'orange', 2: 'b', 3: 'g', 4: 'r', 5: 'darkmagenta', 6: 'chocolate', 7: 'navy', 8: 'limegreen', 9: 'darkred'}
    colors_gen = {0: 'darkmagenta', 1: 'chocolate', 2: 'navy', 3: 'limegreen', 4: 'darkred', 5: 'm', 6: 'orange', 7: 'b', 8: 'g', 9: 'r'}
    
    # Plot states
    for i in range(8):
        ax = fig.add_subplot(5, 2, i+1)
        ax.plot(time, x_real_LQR[i,:], color=colors_ref[i], linestyle='-', linewidth=2, 
                   label=f'{names[i]}')
        ax.plot(time, x_gen[i,:], color=colors_gen[i], linestyle='--', linewidth=2,
                   label=f'{names[i]}' + r'$^{des}$')
        ax.set_title(names[i])

    for i in range(2):
        ax = fig.add_subplot(5, 2, i+9)
        ax.plot(time, u_real_LQR[i,:], color=colors_ref[i+8], linestyle='-', linewidth=2,
                label=f'{names[i+8]}')
        ax.plot(time, u_gen[i,:], color=colors_gen[i+8], linestyle='--', linewidth=2,
                label=f'{names[i+8]}' + r'$^{des}$')
        ax.set_title(names[i+8])
      
    # Adjust layout
    plt.tight_layout()
    plt.show()



def plot_LQR_tracking_errors(x_real_LQR, x_gen, delta_u, t_f=10, dt=1e-4):

    total_time_steps = int(t_f / dt)
    time = np.linspace(0, t_f, total_time_steps)
    
    fig = plt.figure(figsize=(10, 10))
    
    # Define naming and color schemes
    names = {
        0: r'$\dot{z}_1$', 1: r'$\dot{z}_2$', 2: r'$\dot{z}_3$', 3: r'$\dot{z}_4$',
        4: r'$z_1$', 5: r'$z_2$', 6: r'$z_3$', 7: r'$z_4$',
        8: r'$u_1$', 9: r'$u_2$'
    }
    colors = {0: 'm', 1: 'orange', 2: 'b', 3: 'g', 4: 'r', 5: 'darkmagenta', 6: 'chocolate', 7: 'navy', 8: 'limegreen', 9: 'darkred'}
    
    # Plot individual state tracking errors
    for i in range(8):
        error = (x_real_LQR[i,:] - x_gen[i,:])
        ax = fig.add_subplot(5, 2, i+1)
        ax.plot(time, error, color=colors[i], linestyle='-', linewidth=2, 
                   label=f'{names[i]}')
        ax.set_title(names[i])

    for i in range(2):
        error = delta_u[i,:]
        ax = fig.add_subplot(5, 2, i+9)
        ax.plot(time, error, color=colors[i+8], linestyle='-', linewidth=2,
                label=f'{names[i+8]}')
        ax.set_title(names[i+8])
 
    # Adjust layout
    plt.tight_layout()
    plt.show()







