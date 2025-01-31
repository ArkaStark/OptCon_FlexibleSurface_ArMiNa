import numpy as np
import matplotlib.pyplot as plt
from cost_fn import cost
from flexible_dyn import x_next_lambda as dyn_lambda

def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, gradJ,
    xx_ref, uu_ref,  xx, uu, JJ, K_star, sigma_star, iteration, plot = False):

    TT = uu.shape[1]

    stepsizes = []  # list of stepsizes
    costs_armijo = []

    stepsize = stepsize_0
    ns = xx_ref.shape[0]
    ni = uu_ref.shape[0]
    Qt = 10*np.eye(ns)

    Rt = 1e-3*np.ones((ni, ni)) + 1e-4*np.eye(ni)
    QT = 10*np.eye(ns)
    x_next = dyn_lambda()

    descent = 0
    print('grad j 1', gradJ[:,1] )
    print('del u 1', deltau[:,1] )
    for t in range(TT-1):
        descent += np.dot(gradJ[:,t] , deltau[:,t])
        # descent += (gradJ[:,t].T @ deltau[:,t]).flatten()

    print("altmethod:", np.sum(gradJ[:,:-1] * deltau))
    print("descent:", descent)
    for ii in range(armijo_maxiters):

        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = xx[:, 0]

        for tt in range(TT-1):                  
            uu_temp[:,tt] = uu[:,tt] + stepsize*(K_star[:,:,tt] @ (xx_temp[:,tt] - xx[:,tt]) + sigma_star[:,tt])
            xx_temp[:,tt+1] = x_next(xx_temp[:,tt], uu_temp[:,tt]).flatten()

        J_temp = cost(xx_temp, uu_temp, xx_ref, uu_ref,Qt, Rt, QT)
        stepsizes.append(stepsize)      # save the stepsize
        costs_armijo.append(J_temp)    # save the cost associated to the stepsize
        print('armijo jj:',J_temp)
        print('armijo scale factor:', JJ  + cc*stepsize*descent)
        if J_temp > JJ  + cc*stepsize*descent:
            # update the stepsize
            stepsize = beta*stepsize
        else:
            print('Armijo stepsize = {:.3e}'.format(stepsize))
            break

        if ii == armijo_maxiters -1:
            print("WARNING: no stepsize was found with armijo rule!")
            stepsize = 1


    if plot:

        steps = np.linspace(0,stepsize_0,int(2e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):

            step = steps[ii]

            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = xx[:,0]

            for tt in range(TT-1):
                uu_temp[:,tt] = uu[:,tt] + step*(K_star[:,:,tt] @ (xx_temp[:,tt] - xx[:,tt]) + sigma_star[:,tt])
                xx_temp[:,tt+1] = x_next(xx_temp[:,tt], uu_temp[:,tt]).flatten()

            JJ_temp = cost(xx_temp, uu_temp, xx_ref, uu_ref,Qt, Rt, QT)
            costs[ii] = JJ_temp
            # costs[ii] = np.min([JJ_temp.item(), 100 * JJ])


        plt.figure(1)
        plt.clf()


        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
        plt.plot(steps, JJ + descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, JJ + cc*descent*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

        plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()

        plt.show()


    return stepsize
