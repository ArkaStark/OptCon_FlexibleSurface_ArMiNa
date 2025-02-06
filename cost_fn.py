import numpy as np


def cost(x_traj, u_traj, x_ref, u_ref, Qt, Rt, QT):
    
    J=0
    T = x_traj.shape[1]
    for t in range(T-1):
        J = J + stage_cost(x_traj[:, t], x_ref[:, t], u_traj[:, t], u_ref[:, t], Qt, Rt)

    J = J + terminal_cost(x_traj[:, T-1], x_ref[:, T-1], QT)
    # print("J: ", J)
    return J

def stage_cost(x_stage, x_ref, u_stage, u_ref, Qt, Rt):
    delta_x = x_stage - x_ref
    delta_u = u_stage - u_ref

    J_t = 0.5 * delta_x.T @ Qt @ delta_x + 0.5 * delta_u.T @ Rt @ delta_u
    # print("J_t: ", J_t)
    return J_t

def terminal_cost(x_term, x_ref, QT):
    delta_x = x_term - x_ref
    J_T = 0.5 * delta_x.T @ QT @ delta_x
    # print("J_T: ", J_T)
    return J_T

