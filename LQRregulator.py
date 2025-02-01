import numpy as np
from numpy.linalg import inv 
import matplotlib.pyplot as plt
# import parameters as pm
# from parameters import state_perturbation_percentage, affine_perturbation
from flexible_dyn import grad_xu_lambda
from flexible_dyn import x_next_lambda as dyn_lambda
import flexible_dyn as dyn
import cost_fn as cost


state_perturbation_percentage = 0.05
affine_perturbation = 0
x_next = dyn_lambda()
grad_x = grad_xu_lambda()[0]
grad_u = grad_xu_lambda()[1]

def LQR_system_regulator(x_gen, u_gen):

    print('\n\n\
        \t------------------------------------------\n \
        \t\tLaunching: LQR Tracker\n \
        \t------------------------------------------')
    
    x_size = x_gen.shape[0]
    u_size = u_gen.shape[0]
    TT = x_gen.shape[1]

    x_regulator = np.zeros((x_size, TT))
    u_regulator = np.zeros((u_size, TT))
    x_natural_evolution = np.zeros((x_size, TT))
    x_evolution_after_LQR = np.zeros((x_size, TT))

    x_regulator[:, 0]         = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_natural_evolution [:,0] = x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation
    x_evolution_after_LQR[:,0]= x_gen[:,0]*(1 + state_perturbation_percentage) + affine_perturbation

    u_regulator = u_gen

    delta_x = x_regulator - x_gen
    delta_u = u_regulator - u_gen

    # Initialize the perturberd system as the natural evolution of the system
    # without a proper regulation
    for t in range(TT-1):
        x_natural_evolution[:, t+1] = x_next(x_natural_evolution[:, t], u_regulator[:,t]).flatten()


    Qt = np.zeros((x_size, x_size,TT-1))
    Rt = np.zeros((u_size, u_size,TT-1))

    K_Star = np.zeros((u_size, x_size, TT-1))

    A = np.zeros((x_size,x_size,TT-1))
    B = np.zeros((x_size,u_size,TT-1))
    Qt = 10*np.eye(x_size)

    Rt = 1e-3*np.ones((u_size, u_size)) + 1e-4*np.eye(u_size)

    QT = 10*np.eye(x_size)
    for t in range(TT-1):
        A[:,:,t] = grad_x(x_gen[:,t], u_gen[:,t])
        B[:,:,t] = grad_u(x_gen[:,t], u_gen[:,t])


    K_Star = LQR_solver(A, B, Qt, Rt, QT)

    for t in range(TT-1):
        delta_u[:, t]  = K_Star[:,:,t] @ delta_x[:, t]
        delta_x[:, t+1]= A[:,:,t] @ delta_x[:,t] + B[:,:,t] @ delta_u[:, t]
    
    for t in range(TT-1):
        u_regulator[:,t] = u_gen[:,t] + delta_u[:,t]
        x_regulator[:,t] = x_gen[:,t] + delta_x[:,t]
        x_evolution_after_LQR[:,t+1] = x_next(x_evolution_after_LQR[:,t], u_regulator[:,t]).flatten()

    plt.figure()
    for i in range(x_size):
        plt.plot(x_evolution_after_LQR[i, :], color = 'red', label = f'x[{i}]')
    for i in range(u_size):
        plt.plot(u_regulator[i,:], color = 'blue', label = 'u_regulator')
    plt.title("System Evolution with Real Dynamics and LQRegulated input")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    for i in range(x_size):
        plt.plot(delta_x[i, :], color = 'red', label = r'$\Delta$' f'x[{i}]')
    for i in range(u_size):
        plt.plot(delta_u[i,:], color = 'blue', label = r'$\Delta$' 'u')
    plt.title("LQR Residuals evolution")
    plt.legend()
    plt.grid()
    plt.show()

    return x_evolution_after_LQR, delta_u


def LQR_solver(A, B, Qt_Star, Rt_Star, QT_Star):
    
    x_size = A.shape[0]
    u_size = B.shape[1]
    print("b shape 0:", u_size)
    TT = A.shape[2]+1
    
    delta_x = np.zeros((x_size,TT))

    P = np.zeros((x_size,x_size,TT))
    Pt = np.zeros((x_size,x_size))
    Ptt= np.zeros((x_size,x_size))
    
    K = np.zeros((u_size,x_size,TT-1))
    Kt= np.zeros((u_size,x_size))

    ######### Solve the Riccati Equation [S6C4]
    P[:,:,-1] = QT_Star
    Qt  = Qt_Star
    Rt  = Rt_Star

    for t in reversed(range(TT-1)):
        At  = A[:,:,t]
        Bt  = B[:,:,t]
       
        Ptt = P[:,:,t+1]

        temp = (Rt + Bt.T @ Ptt @ Bt)
        inv_temp = -np.linalg.inv(temp)
        Kt = inv_temp @ (Bt.T @ Ptt @ At)
        Pt = At.T @ Ptt @ At + At.T@ Ptt @ Bt @ Kt + Qt

        K[:,:,t] = Kt
        P[:,:,t] = Pt 
    return K
    