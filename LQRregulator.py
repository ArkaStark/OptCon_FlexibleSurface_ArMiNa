import numpy as np
import matplotlib.pyplot as plt
# import parameters as pm
# from parameters import state_perturbation_percentage, affine_perturbation
from flexible_dyn import grad_xu_lambda
from flexible_dyn import x_next_lambda as dyn_lambda

state_perturbation_percentage = 0.02
affine_perturbation = 0
x_next = dyn_lambda()
grad_x = grad_xu_lambda()[0]
grad_u = grad_xu_lambda()[1]

def LQR_system_regulator(x_gen, u_gen):

    print('\n\n\
        \t------------------------------------------\n \
        \t\tLaunching: LQR Tracker\n \
        \t------------------------------------------')
    
    ns = x_gen.shape[0]
    nu = u_gen.shape[0]
    TT = x_gen.shape[1]

    x_regulator = np.zeros((ns, TT))
    u_regulator = np.zeros((nu, TT))
    x_natural_evolution = np.zeros((ns, TT))
    x_LQR_evolution = np.zeros((ns, TT))

    x_regulator[:, 0]         = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_natural_evolution [:,0] = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_LQR_evolution[:,0]= x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation

    u_regulator = u_gen

    delta_x = x_regulator - x_gen
    delta_u = u_regulator - u_gen

    # Initialize the perturberd system as the natural evolution of the system
    # without a proper regulation
    for t in range(TT-1):
        x_natural_evolution[:, t+1] = x_next(x_natural_evolution[:, t], u_regulator[:,t]).flatten()


    Qt = np.zeros((ns, ns,TT-1))
    Rt = np.zeros((nu, nu,TT-1))

    K_Star = np.zeros((nu, ns, TT-1))

    A = np.zeros((ns,ns,TT-1))
    B = np.zeros((ns,nu,TT-1))
    #Qt = 10*np.eye(ns)
    Qt = np.diag(np.array([100] * 4 + [10] * 4))
    Rt = 1e-3*np.ones((nu, nu)) + 1e-4*np.eye(nu)

    QT = 10*np.eye(ns)
    for t in range(TT-1):
        A[:,:,t] = grad_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = grad_u(x_gen[:,t], u_gen[:,t])


    K_Star = LQR_solver(A, B, Qt, Rt, QT)

    for t in range(TT-1):
        delta_u[:, t]  = K_Star[:,:,t] @ delta_x[:, t]
        delta_x[:, t+1]= A[:,:,t] @ delta_x[:,t] + B[:,:,t] @ delta_u[:, t]
        # x_pred = x_next(x_regulator[:, t], u_regulator[:, t]).flatten()
        # delta_x[:, t+1] = x_pred - x_gen[:, t+1]  
    
    for t in range(TT-1):
        u_regulator[:,t] = u_gen[:,t] + delta_u[:,t]
        x_regulator[:,t] = x_gen[:,t] + delta_x[:,t]
        x_LQR_evolution[:,t+1] = x_next(x_LQR_evolution[:,t], u_regulator[:,t]).flatten()

    # plt.figure()
    # for i in range(ns):
    #     plt.plot(x_LQR_evolution[i, :], color = 'red', label = f'x[{i}]')
    # for i in range(nu):
    #     plt.plot(u_regulator[i,:], color = 'blue', label = 'u_regulator')
    # plt.title("System Evolution with Real Dynamics and LQRegulated input")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # plt.figure()
    # for i in range(ns):
    #     plt.plot(delta_x[i, :], color = 'red', label = r'$\Delta$' f'x[{i}]')
    # for i in range(nu):
    #     plt.plot(delta_u[i,:], color = 'blue', label = r'$\Delta$' 'u')
    # plt.title("LQR Residuals evolution")
    # plt.legend()
    # plt.grid()
    # plt.show()

    return x_LQR_evolution, delta_u, x_natural_evolution


def LQR_solver(A, B, Qt_Star, Rt_Star, QT_Star):
    
    ns = A.shape[0]
    nu = B.shape[1]
    TT = A.shape[2]+1
    
    delta_x = np.zeros((ns,TT))

    P = np.zeros((ns,ns,TT))
    Pt = np.zeros((ns,ns))
    Ptt= np.zeros((ns,ns))
    
    K = np.zeros((nu,ns,TT-1))
    Kt= np.zeros((nu,ns))

    ######### Solve the Riccati Equation [S6C4]
    P[:,:,-1] = QT_Star
    Qt  = Qt_Star
    Rt  = Rt_Star

    for t in reversed(range(TT-1)):
        At  = A[:,:,t]
        Bt  = B[:,:,t]
       
        Ptt = P[:,:,t+1]

        temp = (Rt + Bt.T @ Ptt @ Bt)
        inv_temp = -np.linalg.inv(temp + 1e-5 * np.eye(temp.shape[0]))
        Kt = inv_temp @ (Bt.T @ Ptt @ At)
        Pt = At.T @ Ptt @ At + At.T@ Ptt @ Bt @ Kt + Qt

        K[:,:,t] = Kt
        P[:,:,t] = Pt 

    return K
    