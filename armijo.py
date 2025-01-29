import numpy as np
import matplotlib.pyplot as plt
from cost_fn import cost
from flexible_dyn import x_next_lambda as dyn_lambda
 
 
def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, 
                    xx_ref, uu_ref,  x0, uu, JJ, descent_arm, plot = True):
 
      """
      Computes the stepsize using Armijo's rule.
      input parameters:
            - stepsize_0 : initial stepsize guess,
            - armijo_maxiters : maximum number of iterations for armijo rule,
            - deltau : descending direction for the control action,
            - xx_ref : reference trajectory state,
            - uu_ref : reference trajectory input,
            - x0 : initial state,
            - uu : input at current iteration,
            - JJ : cost at current iteration,
            - descent_arm: armijo descent direction at current iteration,
            - plot: whether or not to show descent plot.
 
      output parameters:
            - stepsize
      """
 
      TT = uu.shape[1]
      print('tt:',TT)
 
      stepsizes = []  # list of stepsizes
      costs_armijo = []
 
      stepsize = stepsize_0
 
      ns = xx_ref.shape[0]
      ni = uu_ref.shape[0]
      Qt = 10*np.eye(ns)
 
      Rt = 1e-3*np.ones((ni, ni)) + 1e-4*np.eye(ni)
      QT = 10*np.eye(ns)
      x_next = dyn_lambda()
 
      for ii in range(armijo_maxiters):
 
            # temp solution update
 
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))
 
            xx_temp[:,0] = x0
 
 
            for tt in range(TT-1):
                  uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
                  xx_temp[:,tt+1] = x_next(xx_temp[:,tt], uu_temp[:,tt]).flatten()
 
            # # temp cost calculation
            # JJ_temp = 0
 
            # JJ_temp = cost(xx_temp[:, ii], uu_temp[:, ii], xx_ref, uu_ref, Qt, Rt, QT)
            print("uu_temp:",uu_temp[:, ii])
            print("xx_temp:",xx_temp[:, ii])
            J=0
            T = xx_temp.shape[1]
            for t in range(TT-2):
                J = J + stage_cost(xx_temp[:, t], xx_ref[:, t], uu_temp[:, t], uu_ref[:, t], Qt, Rt)

            J = J + terminal_cost(xx_temp[:, TT-1], xx_ref[:, TT-1], QT)
 
            stepsizes.append(stepsize)      # save the stepsize
            costs_armijo.append(np.min([J, 100*JJ]))    # save the cost associated to the stepsize
            print('armijo jj:',J)
            print('armijo scale factor:', JJ  + cc*stepsize*descent_arm)
            if J > JJ  + cc*stepsize*descent_arm:
                  # update the stepsize
                  stepsize = beta*stepsize
 
            else:
                  print('Armijo stepsize = {:.3e}'.format(stepsize))
                  break
           
            if ii == armijo_maxiters -1:
                  print("WARNING: no stepsize was found with armijo rule!")
           
           
      ############################
      # Descent Plot
      ############################
 
      if plot:
 
            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))
 
            for ii in range(len(steps)):
 
                  step = steps[ii]
 
                  # temp solution update
 
                  xx_temp = np.zeros((ns,TT))
                  uu_temp = np.zeros((ni,TT))
 
                  xx_temp[:,0] = x0
 
                  for tt in range(TT-1):
                        uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                        xx_temp[:,tt+1] = x_next(xx_temp[:,tt], uu_temp[:,tt]).flatten()
 
                        # temp cost calculation
                #   JJ_temp = 0
                #   JJ_temp = cost(xx_temp[:, ii,:], uu_temp[:, ii,:], xx_ref, uu_ref, Qt, Rt, QT)
                  J=0
                  T = xx_temp.shape[1]
                  for t in range(TT-2):
                    J = J + stage_cost(xx_temp[:, t], xx_ref[:, t], uu_temp[:, t], uu_ref[:, t], Qt, Rt)

                  J = J + terminal_cost(xx_temp[:, TT-1], xx_ref[:, TT-1], QT)
 
                  #temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                  #JJ_temp += temp_cost
 
                  costs[ii] = np.min([J, 100*JJ])
 
 
            plt.figure(1)
            plt.clf()
 
     
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            # plt.plot(steps, JJ - descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
 
            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
 
            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()
 
            plt.show()
 
      return stepsize

def stage_cost(x_stage, x_ref, u_stage, u_ref, Qt, Rt):
    delta_x = x_stage - x_ref
    delta_u = u_stage - u_ref

    qt = (Qt @ (x_stage - x_ref)).reshape(-1, 1)
    rt = (Rt @ (u_stage - u_ref)).reshape(-1, 1)

    J_t = qt.T @ delta_x + rt.T @ delta_u + 0.5 * delta_x.T @ Qt @ delta_x + 0.5 * delta_u.T @ Rt @ delta_u
    #print("J_t: ", J_t)
    return J_t[-1]

def terminal_cost(x_term, x_ref, QT):
    delta_x = x_term - x_ref
    qT = (QT@(x_term-x_ref)).reshape(-1, 1)
    J_T = qT.T @ delta_x + 0.5 * delta_x.T @ QT @ delta_x
    #print("J_T: ", J_T)
    return J_T[-1]
 