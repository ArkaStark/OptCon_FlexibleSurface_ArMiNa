import numpy as np
import control as ctrl


from flexible_dyn import grad_xu_lambda
from flexible_dyn import x_next_lambda as dyn_lambda
from cost_fn import cost
from armijo import select_stepsize

def newton_optimal_control(x_ref, u_ref, timesteps=100, task=1, armijo_solver=False):
    
    t = timesteps
    max_iter = 300

    ns = x_ref.shape[0]
    nu = u_ref.shape[0]

    x_opt = np.zeros((ns, t, max_iter+1))
    u_opt = np.zeros((nu, t, max_iter+1))

    K_star = np.zeros((nu, ns, t-1, max_iter))
    sigma_star = np.zeros((nu, t-1, max_iter))
    del_u = np.zeros((nu, t-1, max_iter))

    lmbd = np.zeros((ns, t, max_iter)) # lambdas - costate seq.

    deltau = np.zeros((nu,t, max_iter)) # Du - descent direction
    dJ = np.zeros((nu,t, max_iter))     # DJ - gradient of J wrt u

    lamb = np.zeros((ns, t))
    grad_J_u = np.zeros((nu, t, max_iter))

    JJ = np.zeros(max_iter)      # collect cost
    descent = np.zeros(max_iter) # collect descent direction
    descent_arm = np.zeros(max_iter) # collect descent direction
    visu_descent_plot = False
    A = np.zeros((ns, ns, t-1))
    B = np.zeros((ns, nu, t-1))

    Qt = 10*np.eye(ns)

    Rt = 1e-3*np.ones((nu, nu)) + 1e-4*np.eye(nu)

    St = np.zeros((nu, ns))

    x_next = dyn_lambda()
    grad_x = grad_xu_lambda()[0]
    grad_u = grad_xu_lambda()[1]

    A[:,:,-1] = grad_x(x_ref[:,t-1],u_ref[:,t-1])
    B[:,:,-1] = grad_u(x_ref[:,t-1],u_ref[:,t-1])
    # QT = ctrl.dare(A[:,:,-1], B[:,:,-1], Qt, Rt)[0]
    QT = 10*np.eye(ns)
    # print("QT: ", QT)

    l = np.zeros(max_iter)

    for t in range(t):
        x_opt[:,t, 0] = x_ref[:,0]
        u_opt[:,t, 0] = u_ref[:,0]
    
    print("\n\n\tNewton Optimal Control...\n\n")

    for k in range(max_iter):

        l[k] = cost(x_opt[:, :, k], u_opt[:, :, k], x_ref, u_ref, Qt, Rt, QT)

        # Gradient norm stopping criteria
        if k <= 1:
            print(f"\niteration: {k} \tCost: {l[k]}")
        else: 
            norm_delta_u =  np.linalg.norm(del_u[:,:,k-1])
            print(f"\niteration: {k} \tCost: {l[k]}\tCost reduction: {l[k] - l[k-1]}\tDelta_u Norm: {norm_delta_u}")
            if norm_delta_u < 1e-3:
                break

        # Inutialization of x0 for the next iteration
        x_opt[:,0, k+1] = x_ref[:, 0]

        ##################################
        # Descent direction calculation
        ##################################

        # lmbd_temp = (Qt @ (x_opt[:,t-1,k] - x_ref[:,t-1,k])).reshape(-1, 1)

        # lmbd[:,t-1,k] = lmbd_temp.copy().squeeze()

        for t in range(t-1):
            A[:,:,t] = grad_x(x_opt[:,t,k], u_opt[:,t,k])
            B[:,:,t] = grad_u(x_opt[:,t,k], u_opt[:,t,k])

            # lmbd_temp = A.T@lmbd[:,t+1,k][:,None] + qt       # costate equation
            # dJ_temp = B.T@lmbd[:,t+1,k][:,None] + rt         # gradient of J wrt u
            # deltau_temp = - dJ_temp

            

        K_star[:,:,:,k], sigma_star[:,:,k], del_u[:,:,k] = affine_lqr(x_opt[:,:,k], x_ref, u_opt[:,:,k], u_ref, A, B, Qt, Rt, St, QT)
        # Compute the step sizegi
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
                deltau_temp = - grad_J_u[:,t,k]

                lmbd[:,t,k] = lmbd_temp.squeeze()
                dJ[:,t,k] = dJ_temp.squeeze()
                deltau[:,t,k] = deltau_temp.squeeze()

                descent[k] += deltau[:,t,k].T@deltau[:,t,k]
                descent_arm[k] += dJ[:,t,k].T@deltau[:,t,k]
            
            gamma = select_stepsize( 1,20,  0.5,0.7,
                                deltau[:,:,k], x_ref, u_ref, x_opt[:,0, k+1], 
                                u_opt[:,:,k], JJ[k], descent_arm[k], visu_descent_plot)
            print('gamma:',gamma)
            # armijo(x_opt[:,:,k], x_ref, u_opt[:,:,k], u_ref,
            #                 del_u[:,:,k], grad_J_u[:,:,k], l[k], K_star[:,:,:,k], sigma_star[:,:,k],
            #                 k)
        else:
            gamma = 1

        # Compute the x_opt and u_opt for the next iteration
        for t in range(t-1):
            u_opt[:,t,k+1] = u_opt[:,t,k] + K_star[:,:,t,k] @ (x_opt[:, t, k+1] - x_opt[:, t, k]) + gamma * sigma_star[:,t,k]
            x_opt[:,t+1,k+1] = np.array(x_next(x_opt[:,t,k+1], u_opt[:,t,k+1])).flatten()

    print(f'Algorithm Ended at {k}th iteration')
    return x_opt[:,:,k], u_opt[:,:,k], l

        
def affine_lqr(x_opt, x_ref, u_opt, u_ref, A, B, Qt, Rt, St, QT):
    
    ns = x_opt.shape[0]
    nu = B.shape[1]
    t = x_opt.shape[1]

    del_x = np.zeros((ns, t))
    del_u = np.zeros((nu, t-1))

    del_x[:,0] = x_opt[:,0] - x_ref[:,0]

    ct = np.zeros((ns,1)) 
    Pt = np.zeros((ns,ns))
    Pt= QT # inutially PT

    pt= np.zeros((ns, 1))
    pt=np.diag(QT).reshape(-1,1) # inutially pT

    K = np.zeros((nu,ns,t-1))
    K_t = np.zeros((nu,ns))
    sigma = np.zeros((nu,t-1))
    sigma_t = np.zeros((nu,1))

    for t in reversed(range(t-1)):
        # print("MMt-1: ", np.linalg.eigvals(np.linalg.inv(Rt + B[:,:,t].T @ Pt @ B[:,:,t])))

        rt = (Rt @ (u_opt[:,t] - u_ref[:,t])).reshape(-1, 1)
        qt = (Qt @ (x_opt[:,t] - x_ref[:,t])).reshape(-1, 1)

        K_t = -np.linalg.inv(Rt + B[:,:,t].T @ Pt @ B[:,:,t]) @ (St + B[:,:,t].T @ Pt @ A[:,:,t])

        sigma_t = -np.linalg.inv(Rt + B[:,:,t].T @ Pt @ B[:,:,t]) @ (rt + B[:,:,t].T @ pt +  B[:,:,t].T@Pt@ ct)

        Pt = Qt + A[:,:,t].T @ Pt @ A[:,:,t] - K_t.T @ (Rt + B[:,:,t].T @ Pt @ B[:,:,t]) @ K_t

        pt = qt + A[:,:,t].T @ pt + A[:,:,t].T @ Pt @ ct - K_t.T@(Rt + B[:,:,t].T@Pt@B[:,:,t])@sigma_t

        Pt = Pt
        pt = pt

        K[:,:,t] = K_t
        sigma[:, t] = sigma_t.flatten()

    for t in range(t-1):
        del_u[:,t] = K[:,:,t] @ del_x[:,t] + sigma[:,t]
        del_x[:,t+1] = A[:,:,t] @ del_x[:,t] + B[:,:,t] @ del_u[:,t]

    return K, sigma, del_u
