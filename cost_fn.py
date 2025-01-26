import numpy as np


def cost_task1(x_traj, u_traj, x_ref, u_ref, Qt, Rt, QT):
    
    J=0
    T = x_traj.shape[1]
    for t in range(T-2):
        J = J + stage_cost(x_traj[:, t], x_ref[:, t], u_traj[:, t], u_ref[:, t], Qt, Rt)

    J = J + terminal_cost(x_traj[:, T-1], x_ref[:, T-1], QT)
    return J

def stage_cost(x_stage, x_ref, u_stage, u_ref, Qt, Rt):
    delta_x = x_stage - x_ref
    delta_u = u_stage - u_ref

    qt = np.diag(Qt).reshape(-1, 1)
    rt = np.diag(Rt).reshape(-1, 1)

    J_t = qt.T @ delta_x + rt.T @ delta_u + 0.5 * delta_x.T @ Qt @ delta_x + 0.5 * delta_u.T @ Rt @ delta_u

    return J_t

def terminal_cost(x_term, x_ref, QT):
    delta_x = x_term - x_ref
    qT = np.diag(QT).reshape(-1, 1)
    J_T = qT.T @ delta_x + 0.5 * delta_x.T @ QT @ delta_x

    return J_T

