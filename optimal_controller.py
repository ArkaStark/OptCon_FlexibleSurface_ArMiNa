import numpy as np
import control as ctrl


from flexible_dyn import grad_xu_lambda
from flexible_dyn import x_next_lambda as dyn_lambda
from cost_fn import cost
from armijo import select_stepsize

def newton_optimal_control(x_ref, u_ref, timesteps=100, task=1, armijo_solver=True):
    
    TT = timesteps
    max_iter = 300

    ns = x_ref.shape[0]
    nu = u_ref.shape[0]

    x_opt = np.zeros((ns, TT, max_iter+1))
    u_opt = np.zeros((nu, TT, max_iter+1))

    K_star = np.zeros((nu, ns, TT-1, max_iter))
    sigma_star = np.zeros((nu, TT-1, max_iter))
    del_u = np.zeros((nu, TT-1, max_iter))

    lamb = np.zeros((ns, TT))
    grad_J_u = np.zeros((nu, TT, max_iter))

    A = np.zeros((ns, ns, TT-1))
    B = np.zeros((ns, nu, TT-1))

    Qt = 10*np.eye(ns)

    Rt = 1e-3*np.ones((nu, nu)) + 1e-4*np.eye(nu)

    St = np.zeros((nu, ns))

    x_next = dyn_lambda()
    grad_x = grad_xu_lambda()[0]
    grad_u = grad_xu_lambda()[1]

    lmbd = np.zeros((ns, TT, max_iter))  #lambdas - costate seq.
    deltau = np.zeros((nu, TT, max_iter)) #Du - descent direction
    dJ = np.zeros((nu, TT, max_iter))     #DJ - gradient of J wrt u
#    deltau_temp = np.zeros((nu, TT, max_iter))
    JJ = np.zeros(max_iter)          #collect cost
    descent = np.zeros(max_iter)     #collect descent direction
    descent_arm = np.zeros(max_iter) #collect descent direction
    visu_descent_plot = True

    A[:,:,-1] = grad_x(x_ref[:,TT-1],u_ref[:,TT-1])
    B[:,:,-1] = grad_u(x_ref[:,TT-1],u_ref[:,TT-1])
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
        if k <= 1:
            print(f"\nIteration: {k} \tCost: {l[k]}")
        else: 
            norm_delta_u =  np.linalg.norm(del_u[:,:,k-1])
            print(f"\nIteration: {k} \tCost: {l[k]}\tCost reduction: {l[k] - l[k-1]}\tDelta_u Norm: {norm_delta_u}")
            if norm_delta_u < 1e-3:
                break

        # Initialization of x0 for the next iteration
        x_opt[:,0, k+1] = x_ref[:, 0]

        for t in range(TT-1):
            A[:,:,t] = grad_x(x_opt[:,t,k], u_opt[:,t,k])
            B[:,:,t] = grad_u(x_opt[:,t,k], u_opt[:,t,k])

        K_star[:,:,:,k], sigma_star[:,:,k], del_u[:,:,k] = affine_lqr(x_opt[:,:,k], x_ref, u_opt[:,:,k], u_ref, A, B, Qt, Rt, St, QT)
        # Compute the step size
        if armijo_solver==True and k>0:
            print("TODO")

            ########## Solve the costate equation [S20C5]
            # Compute the effects of the inputs evolution on cost (rt)
            # and on dynamics (B*Lambda)
            lmbd_temp = (QT @ (x_opt[:,-1,k] - x_ref[:,-1]))
            lmbd[:,t-1,k] = lmbd_temp.copy().squeeze()
            for t in reversed(range(t-1)):
                rt = (Rt @ (u_opt[:,t,k] - u_ref[:,t]))
                qt = (Qt @ (x_opt[:,t,k] - x_ref[:,t]))
                lmbd_temp = A[:,:,t].T @ lmbd[:,t+1,k] + qt
                dJ_temp = B[:,:,t].T @ lmbd[:,t+1,k] + rt

                deltau_temp = - dJ_temp
 
                lmbd[:,t,k] = lmbd_temp.squeeze()
                dJ[:,t,k] = dJ_temp.squeeze()
                deltau[:,t,k] = deltau_temp.squeeze()
 
                descent[k] += deltau[:,t,k].T@deltau[:,t,k]
                descent_arm[k] += dJ[:,t,k].T@deltau[:,t,k]
            

            gamma = select_stepsize( 1,20,  0.5,0.7,
                                deltau[:,:,k], x_ref, u_ref, x_opt[:,0, k+1],
                                u_opt[:,:,k], l[k], descent_arm[k], visu_descent_plot)
            print('gamma:',gamma)
 
        else:
            gamma = 1

        # Compute the x_opt and u_opt for the next iteration
        for t in range(TT-1):
            u_opt[:,t,k+1] = u_opt[:,t,k] + K_star[:,:,t,k] @ (x_opt[:, t, k+1] - x_opt[:, t, k]) + gamma * sigma_star[:,t,k]
            x_opt[:,t+1,k+1] = np.array(x_next(x_opt[:,t,k+1], u_opt[:,t,k+1])).flatten()

    print(f'Algorithm Ended at {k}th iteration')
    return x_opt[:,:,k], u_opt[:,:,k], l

        
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
