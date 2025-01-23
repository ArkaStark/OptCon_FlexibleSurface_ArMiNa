import numpy as np
import sympy as sp
import cvxpy as cvx
import scipy as sci
 
# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from solver_ltv_LQR import ltv_LQR

ns = 8
ni = 2

TT = int(5.e1)

# System dynamics - double integrator

AA = np.array([[1,1],[0, 1]])
BB = np.array([[0], [1]])

##############################
# Reference signal
##############################

xx_ref = np.zeros((ns, TT))

# Step reference signal - for first state only
T_mid = int((TT/2))

for tt in range(TT):
  if tt < T_mid:
    xx_ref[0, tt] = 0
  else:
    xx_ref[0, tt] = 10

##############################
# Cost 
##############################

QQ = np.array([[10, 0], [0, 1]])
QQ_f = np.array([[10, 0], [0, 1]])

r = 0.5
RR = r*np.eye(ni)

SS = np.zeros((ni,ns))

# Affine terms (for tracking)

qq = np.zeros((ns,TT))
rr = np.zeros((ni,TT))

for tt in range(TT):
    qq_temp = -QQ@xx_ref[:,tt]
    qq[:,tt] = qq_temp.squeeze()

qqf =  -QQ_f@xx_ref[:,-1]

##############################
# Solver 
##############################

# initial condition
x0 = np.array([0, 0])

KK,sigma = ltv_LQR(AA,BB,QQ,RR,SS,QQ_f, TT, x0, qq, rr, qqf)[:2]

xx = np.zeros((ns, TT))
uu = np.zeros((ni, TT))

for tt in range(TT-1): 
  #
  # Trajectory
  #
  uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
  xx_p = AA@xx[:,tt] + BB@uu[:, tt]
  #
  xx[:,tt+1] = xx_p
  #
