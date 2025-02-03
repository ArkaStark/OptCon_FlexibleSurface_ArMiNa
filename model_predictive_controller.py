import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from flexible_dyn import grad_xu_lambda, x_next_lambda

def mpc(x_gen, u_gen):

    print("\n\n\tModel Predictive Control...\n\n")
    
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    TT = x_gen.shape[1]

    T_pred = 10 # Prediction horizon

    A = np.zeros((x_size, x_size, TT-1))
    B = np.zeros((x_size, u_size, TT-1))

    Qt = 10*np.eye(x_size)
    Rt = 1e-3*np.ones((u_size, u_size)) + 1e-4*np.eye(u_size)
    QT = 10*np.eye(x_size)

    x_mpc = np.zeros((x_size, TT))
    u_mpc = np.zeros((u_size, TT))

    grad_x = grad_xu_lambda()[0]
    grad_u = grad_xu_lambda()[1]
    dyn = x_next_lambda()

    for t in range(TT-1):
        A[:,:,t] = grad_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = grad_u(x_gen[:,t], u_gen[:,t])

    initial_perturbation = 0.05*np.random.randn(x_size)

    x_mpc[:,0] = x_gen[:,0]*(1 + initial_perturbation)

    for t in range(TT-2):

        t_hor = np.minimum(t + np.arange(T_pred), TT-1)
        # t_hor_pred = t_hor[:-1] if len(t_hor) > 1 else t_hor
        
        u_mpc_p, problem = solver_linear_mpc(x_mpc[:, t], x_gen[:, t_hor], u_gen[:, t_hor], A[:,:,t_hor-1], B[:,:,t_hor-1], Qt, Rt, QT, T_pred)
        u_mpc[:, t] = u_mpc_p[:, 0]
        x_mpc[:, t+1] = np.array(dyn(x_mpc[:, t], u_mpc[:, t])).flatten()

        if t % 20 == 0: 
            tracking_error_pos = np.linalg.norm(x_mpc[0:4,t] - x_gen[0:4,t])
            tracking_error_vel = np.linalg.norm(x_mpc[4:8,t] - x_gen[4:8,t])
            print(f"t={t}")
            print(f"Position error={tracking_error_pos:.6f}")
            print(f"Velocity error={tracking_error_vel:.6f}")
            print(f"MPC  position:   z1={x_mpc[0,t]:.4f}, z2={x_mpc[1,t]:.4f}, z3={x_mpc[2,t]:.4f}, z4={x_mpc[3,t]:.4f}")
            print(f"Reference position: z1={x_gen[0,t]:.4f}, z2={x_gen[1,t]:.4f}, z3={x_gen[2,t]:.4f}, z4={x_gen[3,t]:.4f}")
            print(f"MPC  velocity:   z1={x_mpc[4,t]:.4f}, z2={x_mpc[5,t]:.4f}, z3={x_mpc[6,t]:.4f}, z4={x_mpc[7,t]:.4f}")
            print(f"Reference velocity: z1={x_gen[4,t]:.4f}, z2={x_gen[5,t]:.4f}, z3={x_gen[6,t]:.4f}, z4={x_gen[7,t]:.4f}")
            print(f"MPC  input:      u2={u_mpc[0,t]:.2f} u4={u_mpc[1,t]:.2f}")
            print(f"Reference input: u2={u_gen[0,t]:.2f} u4={u_gen[1,t]:.2f}")
            if problem.value is not None:
                print(f"cost={problem.value:.6f}")
            print("---")

    return x_mpc, u_mpc


def solver_linear_mpc(x0, x_ref, u_ref, A, B, Qt, Rt, QT, T_pred):

    ns = x_ref.shape[0]
    nu = u_ref.shape[0]


    x_mpc = cp.Variable((ns, T_pred))
    u_mpc = cp.Variable((nu, T_pred))

    # Constraints
    constr = []
    constr.append(x_mpc[:,0] == x0)
    constr.extend([
        cp.vstack([u_mpc[:, t] for t in range(T_pred-1)]) <= 50, # u_max
        cp.vstack([u_mpc[:, t] for t in range(T_pred-1)]) >= -50, # u_min
        cp.vstack([x_mpc[0:4, t] for t in range(T_pred-1)]) <= 0.01, # z_max
        cp.vstack([x_mpc[0:4, t] for t in range(T_pred-1)]) >= -0.01, # z_min
        cp.vstack([x_mpc[4:8, t] for t in range(T_pred-1)]) <= 0.1, # z_max
        cp.vstack([x_mpc[4:8, t] for t in range(T_pred-1)]) >= -0.1, # z_min
    ])
    # Add dynamics constraints
    for t in range(T_pred-1):
        constr.append(x_mpc[:,t+1] == A[:,:,t] @ x_mpc[:,t] + B[:,:,t] @ u_mpc[:,t])

    # Cost function
    cost = 0

    for i in range(T_pred-t):
        cost += cp.quad_form(x_mpc[:,i] - x_ref[:,t], Qt) + cp.quad_form(u_mpc[:,i] - u_ref[:,t], Rt)

    cost += cp.quad_form(x_mpc[:,T_pred-1] - x_ref[:,T_pred-1], QT)

    problem = cp.Problem(cp.Minimize(cost), constr)

   # Try OSQP first as it's typically faster for MPC problems
    try:
        problem.solve(solver='OSQP', warm_start=True)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return u_mpc.value, problem
    except cp.error.SolverError:
        pass
        
    # Fall back to ECOS if OSQP fails
    try:
        problem.solve(solver='ECOS')
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return u_mpc.value, problem
    except cp.error.SolverError:
        print("Both OSQP and ECOS failed to solve the problem.")
        return np.zeros_like((nu, T_pred)), problem
