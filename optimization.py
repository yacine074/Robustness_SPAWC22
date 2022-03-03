
import sys
import os
import numpy as np

#from gekko import GEKKO
#from packages import *
#from parameters import *
from functions import *


def BF_A(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Pr_max = 10.0, Ps_max = 10.0, Pp = 10.0):
    ''' Bruteforce with QoS constraint'''
    Alpha = np.linspace(0, np.sqrt(1.0), 100)
    Pr = np.linspace(0, np.sqrt(Pr_max), 100)
    Ps = np.linspace(0, np.sqrt(Ps_max), 100)


    A,B,C = np.meshgrid(Alpha, Pr, Ps)
   
    # if QoS constraint respected
    mask = (((Gsp*C**2)+(Grp*B**2))+2*(np.sqrt(Gsp*Grp)*A*C*B)) <= A_(Gpp)

    
    A = A[mask]
    B = B[mask]
    C = C[mask]
    
    SNR1 = FDFR(A, C, Gsr, Gpr) 
    SNR2 = FDF2(A, C, B, Gss, Grs, Gps, Pp)
    SNR = np.minimum(SNR1,SNR2)

    ind = np.argmax(SNR)

    SNR_opt, alpha_opt, pr_opt, ps_opt = SNR[ind], A[ind], B[ind], C[ind]
    c = lambda t: (1/2*np.log2(1+t))
    c_func = np.vectorize(c)
    
    return np.array([[Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, c_func(SNR_opt), alpha_opt**2, pr_opt**2, ps_opt**2]])

'''    
uncomment if need it
def Gekko_A(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Pp = 10.0) :

  m = GEKKO(remote=True)
  # Bounds for the constraints and initialization of the variables Alpha, Pr, Ps
  Alpha = m.Var(0.5,lb=0,ub=Alpha_tilde)
  Ps    = m.Var(1.5,lb=0,ub=Psmax_tilde)
  Pr    = m.Var(0,lb=0,ub=Prmax_tilde)
 
 # QoS constraint
 
  m.Equation(Gsp*Ps**2+Grp*Pr**2+2*(np.sqrt(Gsp*Grp)*Alpha*Ps*Pr) <= A_(Gpp)) 
  
  # Calling the two SNR 

  Func_FDFR = FDFR(Alpha, Ps, Gsr, Gpr)
  Func_FDF2 = FDF2(Alpha, Ps, Pr, Gss, Grs, Gps, Pp)
  Z = m.Var()
  # max(min(SNR1,SNR2))
  m.Maximize(Z)
  m.Equation(Z<=Func_FDFR)
  m.Equation(Z<=Func_FDF2)
  m.solve(disp=False)    # solve
  c = lambda t: (1/2*np.log2(1+t))
  c_func = np.vectorize(c)
  res_Gekko = np.array([[Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps,c_func(Z.value[0]),alpha.value[0]**2,Pr.value[0]**2,Ps.value[0]**2]])

  return  res_Gekko
'''

def BF_A_W_qos(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Pp = 10.0):
    ''' Bruteforce without QoS constraint'''
    Alpha = np.linspace(0, np.sqrt(1.0), 100)
    Pr = np.linspace(0, np.sqrt(Pr_max), 100)
    Ps = np.linspace(0, np.sqrt(Ps_max), 100)

    A,B,C = np.meshgrid(Alpha, Pr, Ps)
  
    
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    
    SNR1 = FDFR(A, C, Gsr, Gpr) 
    SNR2 = FDF2(A, C, B, Gss, Grs, Gps, Pp)
    SNR = np.minimum(SNR1, SNR2)

    ind = np.argmax(SNR)

    SNR_opt, alpha_opt, pr_opt, ps_opt = SNR[ind], A[ind], B[ind], C[ind]
    c = lambda t: (1/2*np.log2(1+t))
    c_func = np.vectorize(c)
    
    return np.array([[Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, c_func(SNR_opt), alpha_opt**2, pr_opt**2, ps_opt**2]])

def SA_MDB_DF(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Pp = 10.0):
  ''' Analyical solution without QoS constraint'''
  # Creation of a numpy array of the same dimension as GRP containing Ps_max
  Ps = np.sqrt(Ps_max)*np.ones(Grp.shape)
  Pr = np.sqrt(Pr_max)*np.ones(Grp.shape)
  
  # SNR1 et SNR2
   
  F1 = FDFR(0, Ps, Gsr, Gpr)
  F2 = FDF2(0, Ps, Pr, Gss, Grs, Gps, Pp)
  # Create an empty array to store the optimal_alpha and optimal SNR values
  alphaOpt = np.zeros(Grp.shape)
  SNR_opt = np.zeros(Grp.shape)

  mask = F1 <= F2
  
  # if F1 <= F2 == False, F2 <= F1==True, bool table
  nmask = np.logical_not(mask)
  
  SNR_opt[mask] = FDFR(0, Ps[mask], Gsr[mask], Gpr[mask]) 
  a = (Gsr[nmask]*Ps[nmask]**2*(Gps[nmask]*Pp+1))
  b = (2*np.sqrt(Grs[nmask]*Gss[nmask])*Ps[nmask]*Pr[nmask]*(Gpr[nmask]*Pp+1))
  
  delta = ((4*Grs[nmask]*Gss[nmask]*Ps[nmask]**2*Pr[nmask]**2*(Gpr[nmask]*Pp+1)**2)+\
  (4*Gsr[nmask]**2*Ps[nmask]**4*(Gps[nmask]*Pp+1)**2)-\
  (4*Gsr[nmask]*Ps[nmask]**2*(Gps[nmask]*Pp+1)*(Gss[nmask]*Ps[nmask]**2+Grs[nmask]*Pr[nmask]**2)*(Gpr[nmask]*Pp+1)))
  alphaOpt[nmask] = (-b+np.sqrt(delta))/(2*a)
 
  #F1 is equal to F2
  #SNR_opt[nmask] = FDFR_BF(alphaOpt[nmask], Ps[nmask], Gsr[nmask])
  SNR_opt[nmask] = FDF2(alphaOpt[nmask], Ps[nmask], Pr[nmask], Gss[nmask], Grs[nmask], Gps[nmask], Pp)
  
  res_analytique = np.stack((Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, C(SNR_opt), alphaOpt**2, Pr**2, Ps**2), axis=1)
  
  return  res_analytique    
    
    


