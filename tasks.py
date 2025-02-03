import numpy as np
import matplotlib.pyplot as plt
import LQRregulator as lqr
import equilibrium_pts as eq
import trajectory as traj
from optimal_controller import newton_optimal_control as noc
import animation

def main():

    tasks_to_run = [2]

    if 1 in tasks_to_run:
        print ("\n\n\t TASK 1 \n\n")

        z0 = [0, 0, 0, 0]
        u_eq1 = [-7, 5]
        u_eq2 = [7, -5]
        z_eq1, u_eq1 = eq.find_equilibrium_points(z0, u_eq1)
        z_eq2, u_eq2 = eq.find_equilibrium_points(z0, u_eq2)

        print("Equilibrium point 1: ", z_eq1, "#", u_eq1)
        print("Equilibrium point 2: ", z_eq2, "#", u_eq2)
        eq.plot_eq_pts([z_eq1, z_eq2])

        print("\nGenerating step trajectory between equilibrium points...\n")

        tf = 1
        dt = 1e-4

        z_ref, u_ref = traj.generate_step_trajectory([z_eq1, z_eq2], [u_eq1, u_eq2], t_f=tf, dt=dt)
        # print("x_ref:\n", z_ref)
        # print("u_ref:\n", u_ref)
        traj.plot_trajectory([z_ref], [u_ref], t_f=tf, dt=dt)
        x_ref = np.append(z_ref, np.zeros((4, z_ref.shape[1])), axis=0)

        timesteps = x_ref.shape[1]
        x_gen, u_gen, l = noc(x_ref, u_ref, timesteps=timesteps, armijo_solver=False)
        
        # Plotting

        plt.plot(l)
        plt.xlabel('Iteration $k$', fontsize=12)
        plt.ylabel(r'$J(u^k)$', fontsize=12)
        plt.title('Cost Evolution', fontsize=12)
        plt.show()

        traj.plot_opt_trajectory(x_gen, u_gen, x_ref, u_ref, t_f=tf, dt=dt)
        animation.animate(x_gen[:,0], u_gen, frames=100)

    if 2 in tasks_to_run:
        print ("\n\n\t TASK 2 \n\n")

        # Generate equilibrium points
        z0 = [0, 0, 0, 0]
        z_eq1, u_eq1 = eq.find_equilibrium_points(z0, [-7, 8])
        z_eq2, u_eq2 = eq.find_equilibrium_points(z0, [-2, -3])
        z_eq3, u_eq3 = eq.find_equilibrium_points(z0, [10, -6])
        z_eq4, u_eq4 = eq.find_equilibrium_points(z0, [3, 2])
        z_eq5, u_eq5 = eq.find_equilibrium_points(z0, [0, 0])
        eq.plot_eq_pts([z_eq1, z_eq2, z_eq3, z_eq4, z_eq5])

        tf = 10
        dt = 1e-4

        # Generate trajectory
        print("\nGenerating smooth trajectory between equilibrium points...\n")
        z_eqs = [z_eq1, z_eq2, z_eq3, z_eq4, z_eq5]
        u_eqs = [u_eq1, u_eq2, u_eq3, u_eq4, u_eq5]
        z_step, u_step = traj.generate_step_trajectory(z_eqs, u_eqs, t_f=tf, dt=dt)
        z_ref, u_ref = traj.generate_smooth_trajectory(z_eqs, u_eqs, t_f=tf, dt=dt)
        traj.plot_trajectory([z_ref, z_step], [u_ref, u_step], t_f=tf, dt=dt)

        x_ref = np.append(z_ref, np.zeros((4, z_ref.shape[1])), axis=0)
        timesteps = x_ref.shape[1]
        x_gen, u_gen, l = noc(x_ref, u_ref, timesteps=timesteps, armijo_solver=False)

        # Plotting

        plt.plot(l)
        plt.xlabel('Iteration $k$', fontsize=12)
        plt.ylabel(r'$J(u^k)$', fontsize=12)
        plt.title('Cost Evolution', fontsize=12)
        plt.show()

        traj.plot_opt_trajectory(x_gen, u_gen, x_ref, u_ref, t_f=tf, dt=dt)
        animation.animate(x_gen[:,0], u_gen, frames=100)

        x_LQR, delta_u, x_natural = lqr.LQR_system_regulator(x_gen, u_gen)
        u_LQR = u_gen + delta_u
        traj.plot_LQR_trajectories(x_LQR, u_LQR, x_gen, u_gen, x_natural, t_f=tf, dt=dt)
        traj.plot_LQR_tracking_errors(x_LQR, x_gen, delta_u,t_f=tf, dt=dt)

    



main()