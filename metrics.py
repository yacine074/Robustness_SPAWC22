# -*- coding: utf-8 -*-
"""metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1unLHNfI1T5TNB9dm3RQVXW3JbeM4JMzc
"""

# -*- coding: utf-8 -*-
"""metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Ed3yzmHdQSNrs6FHf5dzo1KfYQ90aat
"""

import sys
import os

from packages import *
from parameters import *
from functions import *
from loss_function import *

### ### ### ### ### ### ### ### Debit metrics ### ### ### ### ### ### ### ### 

def rate(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Alpha, Pr, Ps) : #name debit_DF == > debit ==> rate
  """
  Function for debit calculation.

  Parameters:
      Grp: 1D Array containing Alpha values.
      Gpp: 1D Array containing Gain between primary transmitter and primary receiver.
      Gsr: 1D Array containing Gain between secondary transmitter and relay.
      Gpr: 1D Array containing Gain between primary transmitter and relay.
      Gss: 1D Array containing Gain between secondary transmitter and secondary receiver.
      Grs: 1D Array containing Gain between relay and secondary receiver.
      Gsp: 1D Array containing Gain between secondary transmitter and primary receiver.
      Gps: 1D Array containing Gain between secondary transmitter and primary receiver.
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.

  Returns:
      Different channel gain, alpha, powers and debit
  """
  
  Rso = np.zeros(Grp.shape)
  
  Rso = np.minimum(C(FDFR(Alpha, Ps, Gsr, Gpr)),C(FDF2(Alpha, Ps, Pr, Pp, Gss, Grs, Gps))) #min F1,F2
  res_analytique = np.stack((Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Rso, Alpha, Pr, Ps), axis=1)
  return res_analytique


def avreage_gap(X, Y):
  """avreage gap between the predicted debit and the obtained debit based on bruteforce"""

  return np.mean(X) - np.mean(Y)

def relative_avreage_gap(X, Y):
  """relative avreage gap between the predicted debit and the obtained debit based on bruteforce"""

  return (np.mean(X) - np.mean(Y))/(np.mean(Y))


def evaluation(pred_debit, y_debit): # not completed
  """
      This function calculates the mean of achievable rate, alpha and powers parameters.

      Parameters:
         pred_debit: Predicted rate.
         y_debit: Ground truth rate.
      
      Returns:
         Mean of Pr, Ps, Alpha, Rs and RMSE, NRMSE, R_Square of Rs (Seconday achievable rate) .

  """
  ps_mean_hat, pr_mean_hat, alpha_mean_hat, rs_mean_hat = np.mean(pred_debit[:,11]), np.mean(pred_debit[:,10]), np.mean(pred_debit[:,9]), np.mean(pred_debit[:,8])
  ps_mean_y, pr_mean_y, alpha_mean_y, rs_mean_y = np.mean(y_debit[:,11]), np.mean(y_debit[:,10]), np.mean(y_debit[:,9]), np.mean(y_debit[:,8])
  rs_rmse = mean_squared_error(pred_debit[:,8], y_debit[:,8], squared=True)
  rs_nrmse = nrmse(pred_debit[:,8], y_debit[:,8])
  rs_r2 = r2_score(pred_debit[:,8], y_debit[:,8])
  res = np.stack((pr_mean_hat, pr_mean_y, ps_mean_hat, ps_mean_y, alpha_mean_hat, alpha_mean_y, rs_mean_hat, rs_mean_y, rs_rmse, rs_nrmse, rs_r2))
  return res
############### Primary Achievable Rate Degradation ##################

def Primary_ARD_Percentage(v_tau): 
  def Primary_ARDP(G, y_out): 
    """
      metrics used on DL model for testing Primary achievable rate degradation percentage .

      Parameters:
        G: Channel gain tensor.
        y_out: Predicted parameter.
      Returns:
        percentage of Primary achievable rate degradation 
    """
    Tau = tf.constant(v_tau, dtype=tf.float32) # ==> Tau 
    
    G = tf.cast(G, dtype='float32')
    y_out = tf.cast(y_out, dtype='float32')
    
    # index retrieval

    Grp_indx, Gpp_indx, Gsr_indx, Gpr_indx, Gss_indx, Grs_indx, Gsp_indx, Gps_indx = [0], [1], [2], [3], [4], [5], [6], [7]
    Alpha_indx, Pr_indx, Ps_indx  = [0], [1], [2]

    # tensors retrieval
    Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps,Alpha, Pr, Ps = tf.gather(G, Grp_indx, axis=1), tf.gather(G, Gpp_indx, axis=1), tf.gather(G, Gsr_indx, axis=1), tf.gather(G, Gpr_indx, axis=1), tf.gather(G, Gss_indx, axis=1), tf.gather(G, Grs_indx, axis=1), tf.gather(G, Gsp_indx, axis=1), tf.gather(G, Gps_indx, axis=1),tf.gather(y_out, Alpha_indx, axis=1), tf.gather(y_out, Pr_indx, axis=1), tf.gather(y_out, Ps_indx, axis=1)
      
    #  Primary Power Creation

    Pp = tf.multiply(tf.ones(tf.shape(Pr), dtype=tf.dtypes.float32),10)

    # Rp : C((Gpp*Pp)/(Grp*Pr**2+Gsp*Ps**2+2*(np.sqrt(Gsp*Grp)*Alpha*Ps*Pr)+1) ==> C(H1/H2), where H1 = Gpp*Pp and H2 = R1+R2+1
    H1 = tf.multiply(Gpp, Pp)

    R1 =  tf.add(tf.multiply(Grp, tf.pow(Pr, 2)),tf.multiply(Gsp,tf.pow(Ps, 2)))

    R2 = tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.multiply(tf.sqrt(tf.multiply(Gsp, Grp)),Alpha),tf.multiply(Ps, Pr)))
    
    H2 = tf.add(R1, tf.add(R2 ,tf.constant(1,dtype=tf.float32)))
    
    SNR_P = tf.divide(H1, H2)
    
    Rp =  tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_P)))

    # Rp_ : C(Gpp*Pp)
    SNR_P_ = tf.multiply(Gpp, Pp)
    Rp_ = tf.multiply(tf.constant(0.5,dtype=tf.float32), log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_P_)))
    # 1 - ratio(Rp, Rp_)
    ARD = tf.subtract(tf.constant(1,dtype=tf.float32), tf.divide(Rp, Rp_))

    #ARD > tau  
    mask_PDD = tf.greater(ARD, Tau)# boolean array 
    
    return mask_PDD
  return Primary_ARDP

def Tau_Violation_Number(Grp, Gpp, Gsp, Alpha, Pr, Ps):
    """
    Parameters:
      Grp: 1D Array containing Channel Gain between relay and primary receiver.
      Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Gsp: 1D Array containing Channel Gain between secondary transmitter and primary receiver.
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.
    Returns:
      Number of violated Tau by the Primary achievable rate degradation
    """
    res = pdd(Grp, Gpp, Gsp , Alpha, Pr, Ps)

    mask_tau = np.greater(res, tau) 

    return np.sum(mask_tau)

def ARD_stats(Grp, Gpp, Gsp, Alpha, Pr, Ps):
  """
    Mean and median of Primary achievable rate degradation.

    Parameters:
      Grp: 1D Array containing Channel Gain between relay and primary receiver.
      Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Gsp: 1D Array containing Channel Gain between secondary transmitter and primary receiver.
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.

    Returns:
      Mean, median for Primary achievable rate degradation
  """
  res = pdd(Grp, Gpp, Gsp, Alpha, Pr, Ps)
  res = res[res>tau]
  with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        mean, median = np.nanmean(res), np.nanmedian(res)
    except RuntimeWarning:
        mean, median = 0, 0 
  return mean, median

def ARD_mean(Grp, Gpp, Gsp, Alpha, Pr, Ps):
  """
    Mean of Primary achievable rate degradation.

    Parameters:
      Grp: 1D Array containing Channel Gain between relay and primary receiver.
      Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Gsp: 1D Array containing Channel Gain between secondary transmitter and primary receiver.
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.

    Returns:
      Mean for Primary achievable rate degradation
  """
  res = pdd(Grp, Gpp, Gsp, Alpha, Pr, Ps)
  res = res[res>tau]
  with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        mean = np.nanmean(res)
    except RuntimeWarning:
        mean = 0
  return mean

def results_analysis(test_set, test_GT,Lambda, Learning_rate): #calling loss function loss_DF or loss_DF_WN ?
    
    predicted_debit = []

    predicted_debit_all = []

    debit_gap, debit_gap_all = np.array([]), []

    pdd_vmax, pdd_vmean = np.array([]), np.array([])

    pdd_vmax_all, pdd_vmean_all = [], []

    viloated_tau, viloated_tau_all = np.array([]), []
    
    ard_mean, ard_mean_all = np.array([]), []

    for ld_k in Lambda.keys():

      for lr_k in Learning_rate.keys():

          model = tf.keras.models.load_model(root_dir+'/lambda = '+ld_k+'/weights/'+ld_k+'.h5', custom_objects={'DF_loss':loss_DF(Lambda,tau),'Primary_Achievable_Rate_Degradation':Primary_Achievable_Rate_Degradation,'Primary_ARDP':Primary_ARD_Percentage,'throughput':Achievable_Rate(tau),'V_Qos':QoS_Violation(tau), "custom_sigmoid":custom_sigmoid})

          ##### Evaluation on test set #####
          predictions = model.predict(test_set)

          # debit calculation for ground truth and predicted test set
          debit_hat_test = rate(test_set[:,0], test_set[:,1], test_set[:,2], test_set[:,3], test_set[:,4], test_set[:,5], test_set[:,6], test_set[:,7], predictions[:,0], predictions[:,1], predictions[:,2])
          debit_true_test = stack_dtest(test_set[:,0], test_set[:,1], test_set[:,2], test_set[:,3], test_set[:,4], test_set[:,5], test_set[:,6], test_set[:,7], test_GT[:,0], np.sqrt(test_GT[:,1]), np.sqrt(test_GT[:,2]), np.sqrt(test_GT[:,3]))

          #predicted debit

          predicted_debit.append(debit_hat_test[:,8]) 

          ###### Gap Acheivable rate ######
          #Avreage debit gap between predicted and Brutefroce debit
          debit_gap = np.append(debit_gap, relative_avreage_gap(debit_hat_test[:,8], debit_true_test[:,8])) #[:,7] : extracting debit from ND array

          ###### Primary degradation Violation ######
          #  Primary degradation percentage, mean and median calculation
          # qos_violation calculation
          tau_violation_count = Tau_Violation_Number(test_set[:,0], test_set[:,1], test_set[:,6], predictions[:,0], predictions[:,1], predictions[:,2]) 

          viloated_tau = np.append(viloated_tau, Tau_Violation_percentage(test_set, tau_violation_count))
          ard_mean = np.append(ard_mean, ARD_mean(test_set[:,0], test_set[:,1], test_set[:,6], predictions[:,0], predictions[:,1], predictions[:,2]))

          # max, mean for primary debit degradation
          pdd_vmax, pdd_vmean = np.append(pdd_vmax, pdd_max(test_set[:,0], test_set[:,1], test_set[:,6], predictions[:,0], predictions[:,1], predictions[:,2])), np.append(pdd_vmean, pdd_mean(test_set[:,0], test_set[:,1], test_set[:,6], predictions[:,0], predictions[:,1], predictions[:,2]))
      # append data from the temporary list to the principal list
      debit_gap_all.append(debit_gap)

      predicted_debit_all.append(predicted_debit)
      ard_mean_all.append(ard_mean)

      viloated_tau_all.append(viloated_tau)
      pdd_vmax_all.append(pdd_vmax)
      pdd_vmean_all.append(pdd_vmean)

      # empty temporary lists
      predicted_debit = []
      debit_gap, viloated_tau = np.array([]), np.array([])

      pdd_vmax, pdd_vmean= np.array([]), np.array([])
      ard_mean = np.array([])
    
    return debit_gap_all, viloated_tau_all, pdd_vmean_all, pdd_vmax_all, ard_mean_all

def Tau_Violation_percentage(xtest, violated_tau_nbr):
  """
    Parameters:
      xtest: 1D Array containing test set.
      violated_tau_nbr: number of tau violated by the .
    Returns:
      Number of violated Tau by the Primary achievable rate degradation
  """
  
  return (violated_tau_nbr/xtest.shape[0])*100


def pdd(Grp, Gpp, Gsp, Alpha, Pr, Ps):
  """
      Parameters:
         Grp: 1D Array containing Alpha values.
         Gpp: 1D Array containing Gain between primary transmitter and primary receiver.
         Gsp: 1D Array containing Gain between secondary transmitter and primary receiver.
         Alpha: 1D Array containing Alpha values.
         Pr: 1D Array containing Power of relay.
         Ps: 1D Array containing Power of secondary network.
      
      Returns:
         primary debit degradation .

  """
  Rp = C((Gpp*Pp)/(Grp*Pr**2+Gsp*Ps**2+2*(np.sqrt(Gsp*Grp)*Ps*Pr*Alpha)+1))
  Rp_max = C(Gpp*Pp)
  ratio_Rp = 1-(Rp/Rp_max)
  return ratio_Rp

def primary_debit_degradation(Grp, Gpp, Gsp , Alpha, Pr, Ps):
  """ Min, Max, Mean, Std of primary debit degradation """
  Rp = C((Gpp*Pp)/(Grp*Pr**2+Gsp*Ps**2+2*(np.sqrt(Gsp*Grp)*Ps*Pr*Alpha)+1))
  Rp_max = C(Gpp*Pp)
  ratio_Rp = 1-(Rp/Rp_max)
  return np.min(ratio_Rp),np.max(ratio_Rp),np.mean(ratio_Rp),np.std(ratio_Rp)

def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return mean_squared_error(actual, predicted, squared=True) / (actual.max() - actual.min())


def pdd_min(Grp, Gpp, Gsp , Alpha, Pr, Ps):

  return np.min(pdd(Grp, Gpp, Gsp , Alpha, Pr, Ps))
  
def pdd_max(Grp, Gpp, Gsp , Alpha, Pr, Ps):

  return np.max(pdd(Grp, Gpp, Gsp , Alpha, Pr, Ps))

def pdd_mean(Grp, Gpp, Gsp , Alpha, Pr, Ps):

  return np.mean(pdd(Grp, Gpp, Gsp , Alpha, Pr, Ps))

def pdd_std(Grp, Gpp, Gsp , Alpha, Pr, Ps):
  
  return np.std(pdd(Grp, Gpp, Gsp , Alpha, Pr, Ps))

def Primary_Achievable_Rate_Degradation(G, y_out):
 
    """
      Metrics used on DL model for Primary achievable rate degradation.
      
      Parameters:
        G: Channel gain tensor.
        y_out: Predicted parameter.
      Returns:
        Primary achievable rate degradation for each samples 
    """
    G = tf.cast(G, dtype='float32')
    y_out = tf.cast(y_out, dtype='float32')
    
    # index retrieval

    Grp_indx, Gpp_indx, Gsp_indx = [0], [1], [6]
    Alpha_indx, Pr_indx, Ps_indx  = [0], [1], [2]

    # tensors retrieval
    Grp, Gpp, Gsp, Alpha, Pr, Ps = tf.gather(G, Grp_indx, axis=1), tf.gather(G, Gpp_indx, axis=1), tf.gather(G, Gsp_indx, axis=1), tf.gather(y_out, Alpha_indx, axis=1), tf.gather(y_out, Pr_indx, axis=1), tf.gather(y_out, Ps_indx, axis=1)
    
    
    #  Pp Creation
    Pp = tf.multiply(tf.ones(tf.shape(Pr), dtype=tf.dtypes.float32),10)
    
    # Rp : C((Gpp*Pp)/(Grp*Pr**2+Gsp*Ps**2+2*(np.sqrt(Gsp*Grp)*Alpha*Ps*Pr)+1) ==> C(H1/H2), where H1 = Gpp*Pp and H2 = R1+R2+1
    H1 = tf.multiply(Gpp, Pp)

    R1 =  tf.add(tf.multiply(Grp, tf.pow(Pr, 2)),tf.multiply(Gsp,tf.pow(Ps, 2)))

    R2 = tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.multiply(tf.sqrt(tf.multiply(Gsp, Grp)),Alpha),tf.multiply(Ps, Pr)))
    
    H2 = tf.add(R1, tf.add(R2 ,tf.constant(1,dtype=tf.float32)))
    
    SNR_P = tf.divide(H1, H2)
    
    Rp =  tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_P)))

    # Rp_ : C(Gpp*Pp)
    SNR_P_ = tf.multiply(Gpp, Pp)
    Rp_ = tf.multiply(tf.constant(0.5,dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_P_)))
    # 1 - ratio(Rp, Rp_)
    RRP = tf.subtract(tf.constant(1,dtype=tf.float32), tf.divide(Rp, Rp_))
    return RRP


def Achievable_Rate(v_tau):
  def throughput(G, y_out):
    """
      Metrics used on DL model for throughput calculation.
      This function will get those parameters as input
      G: Channel gain tensor.
      y_out: Predicted parameter.

      Parameters:
        Lambda : Penalty for QoS
        Tau : degradation factor for the primary network
      Returns:
        Throughput mean 
    """

    Tau = tf.constant(v_tau, dtype=tf.float32) # ==> Tau 
    
    G = tf.cast(G, dtype='float32')
    y_out = tf.cast(y_out, dtype='float32')
    
    # index retrieval

    Grp_indx, Gpp_indx, Gsr_indx, Gpr_indx, Gss_indx, Grs_indx, Gsp_indx, Gps_indx = [0], [1], [2], [3], [4], [5], [6], [7]
    Alpha_indx, Pr_indx, Ps_indx  = [0], [1], [2]

    # tensors retrieval
    Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Alpha, Pr, Ps = tf.gather(G, Grp_indx, axis=1), tf.gather(G, Gpp_indx, axis=1), tf.gather(G, Gsr_indx, axis=1), tf.gather(G, Gpr_indx, axis=1), tf.gather(G, Gss_indx, axis=1), tf.gather(G, Grs_indx, axis=1), tf.gather(G, Gsp_indx, axis=1), tf.gather(G, Gps_indx, axis=1), tf.gather(y_out, Alpha_indx, axis=1), tf.gather(y_out, Pr_indx, axis=1), tf.gather(y_out, Ps_indx, axis=1)
    
    #  Pp Creation
    Pp = tf.multiply(tf.ones(tf.shape(Pr), dtype=tf.dtypes.float32),10)
    
    # SNR1 : (Gsr*(1-Alpha**2)*Ps**2)/(Gpr*Pp+1)
    SNR1 = tf.multiply(Gsr,(tf.multiply(tf.subtract(tf.constant(1,dtype=tf.float32), tf.pow(Alpha, 2)), tf.pow(Ps, 2))))
    SNR1 = tf.divide(SNR1, tf.add(tf.multiply(Gpr,Pp),tf.constant(1,dtype=tf.float32)))
   

    # SNR2 : ((Gss*Ps**2+Grs*Pr**2)+2*(np.sqrt(Grs*Gss)*Alpha*Ps*Pr)) ==> L1+L2/Gps*Pp+1
    L1 = tf.add(tf.multiply(Gss,tf.pow(Ps,2)),tf.multiply(Grs,tf.pow(Pr,2)))

    L2 = tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.multiply(tf.sqrt(tf.multiply(Grs,Gss)),Ps),tf.multiply(Alpha,Pr)))
    
    SNR2 = tf.add(L1,L2)
    
    SNR2 = tf.divide(SNR2, tf.add(tf.multiply(Gps,Pp),tf.constant(1,dtype=tf.float32)))
    
    SNR_opt = tf.minimum(SNR1, SNR2)
    
    #C = 1/2*log2(1+SNR_opt)
    # debit calculation
    Rs = tf.multiply(tf.constant(0.5,dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_opt)))
    return Rs
  return throughput
  
### ### ### ### ### ### ### ###  QoS metrics ### ### ### ### ### ### ### ### 

def QoS_Violation(v_tau): 
  def V_Qos(G, y_out): 
    """
      metrics used on DL model for testing QoS viloation .

      Parameters:
        G: Channel gain tensor.
        y_out: Predicted parameter.
      Returns:
        Number of violated QoS 
    """
    Tau = tf.constant(v_tau, dtype=tf.float32) # ==> Tau 
    
    G = tf.cast(G, dtype='float32')
    y_out = tf.cast(y_out, dtype='float32')
    
    # index retrieval

    Grp_indx, Gpp_indx, Gsr_indx, Gpr_indx, Gss_indx, Grs_indx, Gsp_indx = [0], [1], [2], [3], [4], [5], [6]
    Alpha_indx, Pr_indx, Ps_indx  = [0], [1], [2]

    # tensors retrieval
    Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Alpha, Pr, Ps = tf.gather(G, Grp_indx, axis=1), tf.gather(G, Gpp_indx, axis=1), tf.gather(G, Gsr_indx, axis=1), tf.gather(G, Gpr_indx, axis=1), tf.gather(G, Gss_indx, axis=1), tf.gather(G, Grs_indx, axis=1), tf.gather(G, Gsp_indx, axis=1), tf.gather(y_out, Alpha_indx, axis=1), tf.gather(y_out, Pr_indx, axis=1), tf.gather(y_out, Ps_indx, axis=1)
      
    #  Primary Power Creation

    Pp = tf.multiply(tf.ones(tf.shape(Pr), dtype=tf.dtypes.float32),10)

    ########### QoS ################
      
    # function A' ==> A'(Gpp) : (Gpp*Pp)/((1+(Gpp*Pp))**(1-tau)-1)-1 ==> (Gpp*Pp)/(R1) 
    R1 = tf.add(tf.constant(1, dtype=tf.float32),tf.multiply(Gpp,Pp))
    R1 = tf.pow(R1, tf.subtract(tf.constant(1, dtype=tf.float32),Tau))
    R1 = tf.subtract(R1,tf.constant(1, dtype=tf.float32))

    A_ = tf.subtract(tf.divide(tf.multiply(Gpp,Pp),R1),tf.constant(1, dtype=tf.float32))

    #Qos = (Gsp*Ps**2+Grp*Pr**2+2*np.sqrt(Gsp*Grp)*Alpha*Ps*Pr)-A_/A_
    
    Qos = tf.add(tf.add(tf.multiply(Gsp,tf.pow(Ps,2)),tf.multiply(Grp,tf.pow(Pr,2))), tf.multiply(tf.constant(2,dtype=tf.float32),(tf.multiply(tf.multiply(tf.sqrt(tf.multiply(Gsp,Grp)),Ps),tf.multiply(Alpha,Pr)))))
    Qos = tf.subtract(Qos, A_)
    n_Qos = tf.divide(Qos, A_) # Normalization

    #Qos > 10**-5  
    mask_pr = tf.greater(n_Qos,tf.math.pow(tf.constant(10, dtype=tf.float32),QoS_thresh))# boolean array 
    
    return mask_pr
  return V_Qos

def qos_constraint(Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp):
    """
    Function for testing QoS viloation.

    Parameters:
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.
      Grp: 1D Array containing Channel Gain between relay and primary receiver.
      Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Gss: 1D Array containing Channel Gain between secondarytransmitter and secondary receiver.
      Gsp: 1D Array containing Channel Gain between secondary transmitter and primary receiver.

    Returns:
      Number of violated QoS
    """

    mask_pr = np.greater(QoS_Normalized(Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp),np.float_power(10,QoS_thresh)) 

    return np.sum(mask_pr)

def V_QoS_Hist(title, Lambda_value, Lr_value, Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp):
  """
    Parameters:
      title: Title of the figure.
      Lambda_value: Lambda values.
      Lr_value: Learning rate values.
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.
      Grp: 1D Array containing Channel Gain between relay and primary receiver.
      Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Gss: 1D Array containing Channel Gain between secondarytransmitter and secondary receiver.
      Gsp: 1D Array containing Channel Gain between secondary transmitter and primary receiver.
    Returns:
      Histogram for QoS violation
  """
  sns.set(style='white')
  plt.rcParams["figure.figsize"] = (10,5)

  res = QoS_Normalized(Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp)
  fig, ax = plt.subplots(1) # Creates figure fig and add an axes, ax.
  plt.title('QoS violation for lambda = '+Lambda_value+' and Lr = '+Lr_value,fontweight="bold")
  ax.hist(res, 100, density = True)
  #plt.xlim((-1,3))
  plt.grid()
  plt.yscale('log')
  plt.legend([title], loc='best')
  plt.xlabel('QoS violation percentage', fontsize=12)
  plt.ylabel('Samples', fontsize=12)

  fig.savefig('histogram_'+title+'lambda_'+Lambda_value+'lr_'+Lr_value+'.png')
  fig.show() #Only shows figure 1 and removes it from the "current" stack.
 

def QoS_stats(Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp):
  """
    Mean and median of violated QoS.

    Parameters:
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.
      Grp: 1D Array containing Channel Gain between relay and primary receiver.
      Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Gss: 1D Array containing Channel Gain between secondarytransmitter and secondary receiver.
      Gsp: 1D Array containing Channel Gain between secondary transmitter and primary receiver.
    Returns:
      Mean, median for QoS violation
  """
  res = QoS_Normalized(Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp)
  res = res[res>np.float_power(10, QoS_thresh)]
  with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        mean, median = np.nanmean(res), np.nanmedian(res)
    except RuntimeWarning:
        mean, median = 0, 0 
  return mean, median


def v_qos_percentage(xtest, qc_test):
  return (qc_test/xtest.shape[0])*100
  
### ### ### ### ### ### ### ###  mean of pr, ps, alpha, rs and rmse of rs ### ### ### ### ### ### ### ### 

  
### ### ### ### ### ### ### ###  plots ### ### ### ### ### ### ### ### 

def CDB(x, y, debit_GT):
  sns.set(style='white')
  plt.rcParams["figure.figsize"] = (10,5)
  fig = plt.figure(1)

  colors = ['plum', 'green', 'aqua', 'wheat', 'darkred', 'grey', 'peru', 'purple', 'black', 'red']
  m = ['P','o', 'v', '^', '<', '>', 'H', '8', 'p','s']
  marker_N = 1200
  for j in range(x.shape[0]):
    for k in range(x.shape[1]):
           sns.ecdfplot(data=x[k,j,:], label=r'Lr = '+'$'+y[k].replace('_','^{')+'}$', lw=1, ls=':', marker=m[k],markerfacecolor=colors[k], dash_capstyle='round',color = 'black', markersize=8, markevery=marker_N)

    sns.ecdfplot(data=debit_GT[:,8], label='Bruteforce', lw=1, ls=':', marker='p',markerfacecolor='red', dash_capstyle='round',color = 'black', markersize=8, markevery=marker_N)
  
    plt.grid()
    plt.xlabel("Achievable rate")
    
    plt.title(r'Cumulative distribution function '+"(lambda = "+'$'+list(LD.keys())[j].replace('_','^{')+'}$'+')',fontweight="bold")
    
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3)
    fig.savefig('/content/drive/MyDrive/Colab Notebooks/CodeVF/Results/CDB/CDB_'+list(LD.keys())[j]+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.show()
    plt.pause(1)

def train_evaluation(data, val_data, ylab, x_lim, y_lim, Lambda, filename):
  """
      Parameters:
         data: 1D array contains learning data history.
         data: 1D array contains learning data validation history.
         ylab: y label.
         x_lim: set the x limits of the current axes.
         y_lim : set the y-limits of the current axes.
         Lambda : dictionary that contains the values of lambda as key (str) and values (int)  
         filename : path to store the generated figure.
      Returns:
         Plot for achievable rate, Loss, primary rate degradation and QoS violation evolution.
  """

  sns.set(style='white')
  plt.rcParams["figure.figsize"] = (10, 5)
  plt.grid()

  m_train = ['D','o']
  m_val = ['o','P']
  ls_train = ['solid','dotted']
  m_color = ['plum', 'wheat'] 
  fig = plt.figure(1)

  for i in range (0, len(Lambda)) :
    
      plt.plot(data[i][0][:], label = r'Training set ($\lambda ='+list(Lambda.keys())[i].replace('_','^{')+'}$)',ls = ls_train[i], lw = 1
      , markerfacecolor = m_color[i], dash_capstyle = 'round', color = 'black', marker = m_train[i], markersize = 9, markevery = 50)
      plt.plot(val_data[i][0][:], label = 'Validation set ($\lambda = '+list(Lambda.keys())[i].replace('_','^{')+'}$)', ls='-.', lw = 0.4, marker = m_val[i], markerfacecolor = m_color[i], dash_capstyle = 'round', color = 'black', markersize = 9, markevery = 50)
    
      plt.xlabel("Epochs", fontsize= 20)
      plt.grid()
      plt.ylabel(ylab, fontsize= 20)
      plt.xlim(x_lim)
      plt.ylim(y_lim)
      lgd = plt.legend(loc='best', fontsize= 16)#title="Learning rate"

      plt.xticks(fontsize=16)
      plt.yticks(fontsize=16)
      
  plt.show()
  plt.ion()
  fig.savefig(filename+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def test_evaluation(Secondary_rate_gap, Outage, Pnd_average, Pnd_maximum, Delta_out):
    
    # First part : Average and maximum primary rate degradation and average degradation when in outage (∆out) as functions of λ over the test set.
    
    references = np.round(np.array([10**-1,10**-0.75,10**-0.5,10**-0.25,10**0,10**0.25,10**0.5,10**0.75,10**1,10**1.25,10**1.5,10**1.75,10**2]), decimals=4)
   
    rate_gap_plt = np.round(np.array(Secondary_rate_gap).transpose().reshape(references.shape[0]), decimals=4)

    outage_plt = np.round(np.array(Outage).transpose().reshape(references.shape[0]), decimals=4)

    pnd_average_plt = np.round(np.array(Pnd_average).transpose().reshape(references.shape[0]), decimals=4)

    pnd_max_plt = np.round(np.array(Pnd_maximum).transpose().reshape(references.shape[0]), decimals=4)

    delta_out_plt = np.round(np.array(Delta_out).transpose().reshape(references.shape[0]), decimals=4)
    
    plt.rcParams["figure.figsize"] = (10,5)
    sns.set(style='white')
    fig = plt.figure(1)

    plt.grid()
    xs = np.linspace(1, 21, 100)

    plt.hlines(y=25, xmin=0, xmax=len(xs), colors='black', linestyles='--', lw=2, label=r'$\tau = 25\%$')

    plt.plot(references, pnd_average_plt*100, label = 'Average', marker='8',markersize=9)

    plt.plot(references, pnd_max_plt*100, label= 'Max', marker='^',markersize=9)

    plt.plot(references, delta_out_plt*100, label= r'$\Delta_{out}$', marker='s',markersize=9)

    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'Hyperparameter $\lambda$',fontsize= 20)
    plt.ylabel('Primary network degradation (%)',fontsize= 20)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    #plt.xticks(references,['$10^{-1}$', '$10^{-0.75}$','$10^{-0.5}$', '$10^{-0.25}$','$10^{0}$', '$10^{0.25}$','$10^{0.5}$', '$10^{0.75}$','$10^{1}$','$10^{1.25}$','$10^{1.5}$','$10^{1.75}$','$10^{2}$'])
    plt.legend(loc = 'best', fontsize= 16)
    fig.savefig('Primary_network_degradation_stats.pdf', bbox_inches='tight')
    plt.show()

    plt.close()
    
    # Second part : Plot G and Outage
    
    plt.rcParams["figure.figsize"] = (10,5)
    sns.set(style='white')
    fig = plt.figure(1)
    plt.grid()
    
    xs = np.linspace(1, 21, 100)

    plt.plot(references, rate_gap_plt*100, label='Relative average gap', marker='^',markersize=9)
    plt.plot(references, outage_plt, label = 'Outage', marker='H',markersize=9)

    plt.ylabel('Percentage', fontsize= 20)
    plt.xscale('log')
    plt.xlabel(r'Hyperparameter $\lambda$',fontsize= 20)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 

    #plt.xticks(A,['$10^{-1}$', '$10^{-0.75}$','$10^{-0.5}$', '$10^{-0.25}$','$10^{0}$', '$10^{0.25}$','$10^{0.5}$', '$10^{0.75}$','$10^{1}$','$10^{1.25}$','$10^{1.5}$','$10^{1.75}$','$10^{2}$'])
    plt.legend(loc = 'best', fontsize= 16)
    fig.savefig('G_and_Outage.pdf', bbox_inches='tight')
    plt.show()


def Pdd_Hist(Grp, Gpp, Gsp, Alpha, Pr, Ps, Alpha2, Pr2, Ps2):
  """
    Parameters:
      Grp: Channel Gain between relay and primary receiver.
      Gpp: Channel Gain between primary transmitter and primary receiver. 
      Gsp: Channel Gain between secondary transmitter and primary receiver.
      Alpha: Array containing Alpha values.
      Pr: Array containing Power of relay.
      Ps: Array containing Power of secondary network.
    Returns:
      histogram for Primary debit degradation 
  """
  sns.set(style='white')
  plt.rcParams["figure.figsize"] = (10,5)
  
  res = pdd(Grp, Gpp, Gsp ,Alpha, Pr, Ps)*100
  res2 = pdd(Grp, Gpp, Gsp ,Alpha2, Pr2, Ps2)*100

  #fig, ax = plt.subplots(1) # Creates figure fig and add an axes, ax.
  xs = np.linspace(1, 21, 10**5)

  plt.vlines(x=25, ymin=0, ymax=len(xs), colors='black', linestyles='--', lw=2, label=r'$\tau = 25\%$')


  plt.hist(res, 100, histtype='step', ls=':', lw = 2 , color='red',label='$\lambda = 10^{0.5}$')
  plt.hist(res2, 100, histtype='step', ls='-',  lw = 2, label='$\lambda = 10^{2}$')
  
  fig = plt.figure(1)
  #plt.xlim((-1,40))
  plt.grid()
  plt.yscale('log')
  plt.xlabel('Primary acheivable rate degradation ($\%$)', fontsize= 20)
  plt.ylabel('Samples', fontsize= 20)
  plt.legend(loc='best')
  #plt.annotate(r"$\lambda$ = "+'$'+Lambda_value.replace('_','^{')+'}$', xy=(0.05,0.9),xycoords='axes fraction',
  #           fontsize=14)

  lgd = plt.legend(loc='best',
        fancybox=True, shadow=True, fontsize= 16)#title="Learning rate"
  
  plt.xticks(fontsize=16) 
  plt.yticks(fontsize=16) 

  fig.savefig('_vf''.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.show()
  plt.ion()
  plt.pause(1)
  plt.close()  
    
def results_grid(title, data, lambda_label, lr_label):
  df = pd.DataFrame(data, lambda_label.keys(), lr_label.keys())
  fig, ax = plt.subplots(figsize=(10,10))
  plt.title(title)
  with sns.axes_style('white'):
    ax = sns.heatmap(df, annot=True, linewidths=.5, ax=ax, cmap=ListedColormap(['beige']))
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    plt.savefig(title, bbox_inches='tight')

