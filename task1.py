import numpy as np
import equilibrium_pts as eq
import reference_trajectory as ref

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

    x_ref, u_ref = ref.generate_trajectory(z_eq1, z_eq2, u_eq1, u_eq2)
    print("x_ref:\n", x_ref)
    print("u_ref:\n", u_ref)
    ref.plot_trajectory(x_ref, u_ref)

main()