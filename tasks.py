import numpy as np
import matplotlib.pyplot as plt
import os

import equilibrium_pts as eq
import trajectory as traj
from newton_optimal_controller import noc
from linear_quadratic_regulator import lqr
from model_predictive_controller import mpc
import animation

def main():

    tf = 3
    dt = 1e-4

    tasks_to_run = [3]

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

        z_ref, u_ref = traj.generate_step_trajectory([z_eq1, z_eq2], [u_eq1, u_eq2], t_f=tf, dt=dt)
        # print("x_ref:\n", z_ref)
        # print("u_ref:\n", u_ref)
        traj.plot_trajectory([z_ref], [u_ref], t_f=tf, dt=dt)
        x_ref = np.append(z_ref, np.zeros((4, z_ref.shape[1])), axis=0)

        timesteps = x_ref.shape[1]
        x_gen, u_gen, l = noc(x_ref, u_ref, timesteps=timesteps, armijo_solver=False)
        
        # Plotting

        plt.semilogy(l)
        plt.xlabel('Iteration $k$', fontsize=12)
        plt.ylabel(r'$J(u^k)$', fontsize=12)
        plt.title('Cost Evolution (Semi-Logarithmic Scale)', fontsize=12)
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
        np.savez('optimal_trajectory.npz', x_gen=x_gen, u_gen=u_gen)

        # Plotting

        plt.semilogy(l)
        plt.xlabel('Iteration $k$', fontsize=12)
        plt.ylabel(r'$J(u^k)$', fontsize=12)
        plt.title('Cost Evolution (Semi-Logarithmic Scale)', fontsize=12)
        plt.show()

        traj.plot_opt_trajectory(x_gen, u_gen, x_ref, u_ref, t_f=tf, dt=dt)
        animation.animate(x_gen[:,0], u_gen, frames=100)

    if 3 in tasks_to_run:
            print ("\n\n\t TASK 3 \n\n")

            recompute = True
            if recompute==False and os.path.exists('lqr_results.npz'):
                print("\nLoading LQR results...\n")
                x_lqr, u_lqr = np.load('lqr_results.npz')['x_lqr'], np.load('lqr_results.npz')['u_lqr']
            else:
                x_gen, u_gen = np.load('optimal_trajectory.npz')['x_gen'], np.load('optimal_trajectory.npz')['u_gen']
                x_lqr, u_lqr = lqr(x_gen, u_gen)
                np.savez('lqr_results.npz', x_lqr=x_lqr, u_lqr=u_lqr)

            traj.plot_opt_trajectory(x_lqr, u_lqr, x_gen, u_gen, t_f=tf, dt=dt)
            traj.plot_tracking_error(x_lqr, u_lqr, x_gen, u_gen, t_f=tf, dt=dt)
            animation.animate(x_lqr[:,0], u_lqr, frames=100)

    if 4 in tasks_to_run:
        print ("\n\n\t TASK 4 \n\n")

        recompute = True
        if recompute==False and os.path.exists('mpc_results.npz'):
            print("\nLoading MPC results...\n")
            x_mpc, u_mpc = np.load('mpc_results.npz')['x_mpc'], np.load('mpc_results.npz')['u_mpc']
        else:
            x_gen, u_gen = np.load('optimal_trajectory.npz')['x_gen'], np.load('optimal_trajectory.npz')['u_gen']
            x_mpc, u_mpc = mpc(x_gen, u_gen)
            np.savez('mpc_results.npz', x_mpc=x_mpc, u_mpc=u_mpc)

        traj.plot_opt_trajectory(x_mpc, u_mpc, x_gen, u_gen, t_f=tf, dt=dt)
        traj.plot_tracking_error(x_mpc, u_mpc, x_gen, u_gen, t_f=tf, dt=dt)
        animation.animate(x_mpc[:,0], u_mpc, frames=100)




main()