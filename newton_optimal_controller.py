import numpy as np
import control as ctrl


from flexible_dyn import grad_xu_lambda
from flexible_dyn import x_next_lambda as dyn_lambda
from cost_fn import cost
from armijo import armijo
from trajectory import plot_opt_trajectory

def noc(x_ref, u_ref, timesteps=100, armijo_solver=False):

    plot_intermediate_traj = False
    
    TT = timesteps
    max_iter = 300

    ns = x_ref.shape[0]
    nu = u_ref.shape[0]

    x_opt = np.zeros((ns, TT, max_iter+1))
    u_opt = np.zeros((nu, TT, max_iter+1))

    K_star = np.zeros((nu, ns, TT-1, max_iter))
    sigma_star = np.zeros((nu, TT-1, max_iter))
    del_u = np.zeros((nu, TT-1, max_iter))
    norm_delta_u = np.zeros(max_iter)

    lamb = np.zeros((ns, TT))
    grad_J_u = np.zeros((nu, TT-1, max_iter))

    A = np.zeros((ns, ns, TT-1))
    B = np.zeros((ns, nu, TT-1))

    Qt = 10*np.eye(ns)
    Rt = 1e-4*np.eye(nu)
    St = np.zeros((nu, ns))

    x_next = dyn_lambda()
    grad_x = grad_xu_lambda()[0]
    grad_u = grad_xu_lambda()[1]

    A[:,:,-1] = grad_x(x_ref[:,TT-1],u_ref[:,TT-1])
    B[:,:,-1] = grad_u(x_ref[:,TT-1],u_ref[:,TT-1])
    # print("A:", A[:, :, -1])
    # print("B: ", B[:, :, -1])
    # QT = ctrl.dare(A[:,:,-1], B[:,:,-1], Qt, Rt)[0]
    QT = 10*np.eye(ns)
    # print("QT: ", QT)

    l = np.zeros(max_iter)

    for t in range(TT):
        x_opt[:,t, 0] = x_ref[:,0]
        u_opt[:,t, 0] = u_ref[:,0]
    
    print("\n\n\tNewton Optimal Control...\n\n")

    for k in range(max_iter):

        l[k] = cost(x_opt[:, :, k], u_opt[:, :, k], x_ref, u_ref, Qt, Rt, QT)

        # Gradient norm stopping criteria
        if k < 1:
            print(f"\nIteration: {k} \tCost: {l[k]}")
        else: 
            norm_delta_u[k] =  np.linalg.norm(del_u[:,:,k-1])
            print(f"\nIteration: {k} \tCost: {l[k]}\tCost reduction: {l[k] - l[k-1]}\tDelta_u Norm: {norm_delta_u[k]}")
            if norm_delta_u[k] < 1e-3:
                break

        # Initialization of x0 for the next iteration
        x_opt[:,0, k+1] = x_ref[:, 0]

        for t in range(TT-1):
            A[:,:,t] = grad_x(x_opt[:,t,k], u_opt[:,t,k])
            B[:,:,t] = grad_u(x_opt[:,t,k], u_opt[:,t,k])

        K_star[:,:,:,k], sigma_star[:,:,k], del_u[:,:,k] = affine_lqr(x_opt[:,:,k], x_ref, u_opt[:,:,k], u_ref, A, B, Qt, Rt, St, QT)

        # Compute the step size
        if armijo_solver==True and k>0:
            print("Running Armijo...")
            qT = (QT @ (x_opt[:,-1,k] - x_ref[:,-1]))
            lamb[:,-1] = qT
            for t in reversed(range(TT-1)):
                rt = (Rt @ (u_opt[:,t,k] - u_ref[:,t]))
                qt = (Qt @ (x_opt[:,t,k] - x_ref[:,t]))
                lamb[:,t] = A[:,:,t].T @ lamb[:,t+1] + qt
                grad_J_u[:,t,k] = B[:,:,t].T @ lamb[:,t+1] + rt
            
            gamma = armijo(x_opt[:,:,k], x_ref, u_opt[:,:,k], u_ref, 
                                  del_u[:,:,k], grad_J_u[:,:,k], l[k], K_star[:,:,:,k], sigma_star[:,:,k], Qt, Rt, QT, k, plot=True)
            # print('gamma:',gamma)
        else:
            gamma = 0.1
            print('selected step_size:', gamma)

        # Compute the x_opt and u_opt for the next iteration
        for t in range(TT-1):
            u_opt[:,t,k+1] = u_opt[:,t,k] + K_star[:,:,t,k] @ (x_opt[:, t, k+1] - x_opt[:, t, k]) + gamma * sigma_star[:,t,k]
            x_opt[:,t+1,k+1] = np.array(x_next(x_opt[:,t,k+1], u_opt[:,t,k+1])).flatten()

        if plot_intermediate_traj and k%3==0:
            plot_opt_trajectory(x_opt[:,:,k], u_opt[:,:,k], x_ref, u_ref, t_f=TT, dt=1)

    print(f'Algorithm Ended at {k}th iteration')
    return x_opt[:,:,k], u_opt[:,:,k], l, norm_delta_u

        
def affine_lqr(x_opt, x_ref, u_opt, u_ref, A, B, Qt, Rt, St, QT):
    
    ns = x_opt.shape[0]
    nu = B.shape[1]
    TT = x_opt.shape[1]

    del_x = np.zeros((ns, TT))
    del_u = np.zeros((nu, TT-1))

    del_x[:,0] = x_opt[:,0] - x_ref[:,0]

    ct = np.zeros((ns,1)) 
    Pt = np.zeros((ns,ns))
    Ptt= QT # initially PT

    pt= np.zeros((ns, 1))
    ptt=np.diag(QT).reshape(-1,1) # initially pT

    K = np.zeros((nu,ns,TT-1))
    K_t = np.zeros((nu,ns))
    sigma = np.zeros((nu,TT-1))
    sigma_t = np.zeros((nu,1))

    for t in reversed(range(TT-1)):
        # print("MMt-1: ", np.linalg.eigvals(np.linalg.inv(Rt + B[:,:,t].T @ Ptt @ B[:,:,t])))

        rt = (Rt @ (u_opt[:,t] - u_ref[:,t])).reshape(-1, 1)
        qt = (Qt @ (x_opt[:,t] - x_ref[:,t])).reshape(-1, 1)

        K_t = -np.linalg.inv(Rt + B[:,:,t].T @ Ptt @ B[:,:,t]) @ (St + B[:,:,t].T @ Ptt @ A[:,:,t])

        sigma_t = -np.linalg.inv(Rt + B[:,:,t].T @ Ptt @ B[:,:,t]) @ (rt + B[:,:,t].T @ ptt +  B[:,:,t].T@Ptt@ ct)

        Pt = Qt + A[:,:,t].T @ Ptt @ A[:,:,t] - K_t.T @ (Rt + B[:,:,t].T @ Ptt @ B[:,:,t]) @ K_t

        pt = qt + A[:,:,t].T @ ptt + A[:,:,t].T @ Ptt @ ct - K_t.T@(Rt + B[:,:,t].T@Ptt@B[:,:,t])@sigma_t

        Ptt = Pt
        ptt = pt

        K[:,:,t] = K_t
        sigma[:, t] = sigma_t.flatten()

    for t in range(TT-1):
        del_u[:,t] = K[:,:,t] @ del_x[:,t] + sigma[:,t]
        del_x[:,t+1] = A[:,:,t] @ del_x[:,t] + B[:,:,t] @ del_u[:,t]

    return K, sigma, del_u
