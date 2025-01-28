import numpy as np


def cost(x_traj, u_traj, x_ref, u_ref, Qt, Rt, QT):
    
    J=0
    T = x_traj.shape[1]
    for t in range(T-2):
        J = J + stage_cost(x_traj[:, t], x_ref[:, t], u_traj[:, t], u_ref[:, t], Qt, Rt)
    #stageCost = J
    terminalCost = terminal_cost(x_traj[:, T-1], x_ref[:, T-1], QT)
    J = J + terminalCost
    # print("J: ", J)
    return J

def stage_cost(x_stage, x_ref, u_stage, u_ref, Qt, Rt):
    delta_x = x_stage - x_ref
    delta_u = u_stage - u_ref

    qt = (Qt @ (x_stage - x_ref)).reshape(-1, 1)
    rt = (Rt @ (u_stage - u_ref)).reshape(-1, 1)

    J_t = qt.T @ delta_x + rt.T @ delta_u + 0.5 * delta_x.T @ Qt @ delta_x + 0.5 * delta_u.T @ Rt @ delta_u
    # print("J_t: ", J_t)
    return J_t[-1]

def terminal_cost(x_term, x_ref, QT):
    delta_x = x_term - x_ref
    qT = (QT@(x_term-x_ref)).reshape(-1, 1)
    J_T = qT.T @ delta_x + 0.5 * delta_x.T @ QT @ delta_x
    # print("J_T: ", J_T)
    return J_T[-1]

