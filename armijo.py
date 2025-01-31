import numpy as np
import matplotlib.pyplot as plt

from flexible_dyn import x_next_lambda as dyn_lambda
from cost_fn import cost

def armijo(x_trajectory, x_reference, u_trajectory, u_reference, delta_u, gradJ, J, Kt, sigma_t, Qt, Rt, QT, arm_max_iter=20, step_size_0=1, c = 0.5, beta = 0.7, plot = False):
    """
    Perform the Armijo backtracking line search to find an appropriate step size for the optimization.

    Args:
        x_trajectory (ndarray): Current state trajectory.
        x_reference (ndarray): Reference state trajectory.
        u_trajectory (ndarray): Current control trajectory.
        u_reference (ndarray): Reference control trajectory.
        delta_u (ndarray): Control update.
        gradJ (ndarray): Gradient of the cost function.
        J (float): Current cost function value.
        Kt (ndarray): Gain matrix for feedback control.
        sigma_t (ndarray): Scaling factor for the step size.
        iteration (int): Current iteration number.
        task (int, optional): Task identifier (default is 1).
        step_size_0 (float, optional): Initial step size for the search (default is 0.1).

    Returns:
        float: The step size selected by the Armijo method.
    """

    x_next = dyn_lambda()
    
    x_size = x_reference.shape[0]
    horizon = x_reference.shape[1]

    # resolution for plotting the cost function
    resolution = 20   
    
    ## Initialize the following variables:
    #  -    step_size: gamma that is considered at the i-th iteration
    #  -    gamma: from 1 to resolution, used to generate the plots
    #  -    step_sizes: vector of all the step_sizes evaluated until the i-th iteration
    #  -    costs_armijo: vector of the costs of the function evaluated until the i-th iteration
    
    step_size = step_size_0
    gamma = np.linspace(0, 1, resolution)
    step_sizes = []
    costs_armijo = []

    x_size = x_reference.shape[0]
    u_size = u_reference.shape[0]

    x_update = np.zeros((x_size,horizon))
    u_update = np.zeros((u_size,horizon))

    descent = 0
    for t in range(horizon-1):
        # print(f'gradJ = {gradJ[:,t]}')
        # print(f'delta_u = {delta_u[:,t]}')
        # print(gradJ[:,t].shape)
        # print(delta_u[:,t].shape)
        # print(gradJ[:,t].T @ delta_u[:,t])
        descent = descent + gradJ[:,t].T @ delta_u[:,t]
        # descent = descent - gradJ[:,t].T @ gradJ[:,t]
        # descent = descent - delta_u[:,t].T @ delta_u[:,t]

    print(f'descent = {descent}')

    for i in range(arm_max_iter-1):
        x_update[:,:] = x_trajectory
        u_update[:,:] = u_trajectory
        for t in range(horizon-1):
            u_update[:,t] = u_trajectory[:,t] + (Kt[:,:,t] @ (x_update[:,t] - x_trajectory[:,t]) + sigma_t[:,t]) * step_size
            x_update[:,t+1] = np.array(x_next(x_update[:,t], u_update[:,t])).flatten()

        J_temp = cost(x_update, u_update, x_reference, u_reference, Qt, Rt, QT)

        step_sizes.append(step_size)
        costs_armijo.append(J_temp)

        if (J_temp > J + c * step_size * descent):
            #print('J_temp = {}'.format(J_temp))
            step_size = beta * step_size
            if i == arm_max_iter-2:
                print(f'Armijo method did not converge in {arm_max_iter} iterations')
                step_size = 0.001
                break

        else:
            print(f'Selected Armijo step_size = {step_size}')
            break    

    if plot == True: #and ((iteration <= Arm_plot_up_to_iter_k)or(iteration >= Arm_plot_from_iter_k_to_end ))  and iteration!=0:
        # Armijo Plot
        x_temp_sec = np.zeros((x_size, horizon, resolution))
        u_temp_sec = np.zeros((u_size, horizon, resolution))
        J_plot = np.zeros(resolution)
   
        for j in range(resolution):
            x_temp_sec[:,:,j] = x_trajectory
            u_temp_sec[:,:,j] = u_trajectory
   
        for j in range(resolution):
            for t in range(horizon-1):
                u_temp_sec[:,t,j] = u_trajectory[:,t] + (Kt[:,:,t] @ (x_temp_sec[:,t,j] - x_trajectory[:,t]) + sigma_t[:,t]) * gamma[j]
                x_temp_sec[:,t+1,j] = np.array(x_next(x_temp_sec[:,t,j], u_temp_sec[:,t,j])).flatten()
                
            J_plot[j] = cost(x_temp_sec[:,:,j], u_temp_sec[:,:,j], x_reference, u_reference, Qt, Rt, QT)

        plt.plot(gamma, J+c*gamma*descent, color='red', label='Armijo Condition')
        plt.plot(gamma, J+gamma*descent, color='black', label='Tangent Line')
        plt.plot(gamma, J_plot, color='green', label='Cost Evolution')
        plt.scatter(step_sizes, costs_armijo, color='blue', label='Armijo Steps')
        plt.grid()
        plt.legend()
        plt.show()

    return step_size