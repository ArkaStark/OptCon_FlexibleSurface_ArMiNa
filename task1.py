import numpy as np
import equilibrium_pts as eq
import trajectory as traj
from optimal_controller import newton_optimal_control as noc

def main():
    z0 = [0, 0, 0, 0]
    u_eq1 = [-200, 500]
    u_eq2 = [200, -500]
    z_eq1 = eq.find_equilibrium_points(z0, u_eq1)
    z_eq2 = eq.find_equilibrium_points(z0, u_eq2)

    print("Equilibrium point 1: ", z_eq1)
    print("Equilibrium point 2: ", z_eq2)
    eq.plot_eq_pts(z_eq1, z_eq2)

    print("Generating trajectory between equilibrium points...")

    tf = 0.01
    dt = 1e-4

    z_ref, u_ref = traj.generate_trajectory(z_eq1, z_eq2, u_eq1, u_eq2, t_f=tf, dt=dt)
    # print("x_ref:\n", z_ref)
    # print("u_ref:\n", u_ref)
    traj.plot_trajectory(z_ref, u_ref, t_f=tf, dt=dt)

    x_ref = np.append(z_ref, np.zeros((4, z_ref.shape[1])), axis=0)

    timestep = x_ref.shape[1]
    x_gen, u_gen, l = noc(x_ref, u_ref, timesteps=timestep, task=1, armijo_solver=True)
    traj.plot_opt_trajectory(x_gen, u_gen, x_ref, u_ref, t_f=tf, dt=dt)

main()