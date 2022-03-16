"""
  this file contains the different useful functions.

"""

#from packages import *
#from parameters import *

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn
from multiprocessing import Pool

from utils import *

from optimization import *







def A_(Gpp, Pp=10.0, tau=0.25):
    """
      Compute the QoS (A).
      Parameters:
         Gpp: 1D Array containing Channel Gain between primary transmitter and primary receiver.
      Returns:
        A' results.
    """  
    return ((Gpp*Pp)/((1+(Gpp*Pp))**(1-tau)-1))-1
    
def C(x):
    """
      Shannon function.
      Parameters:
         x: Signal-to-noise-ratio.
      Returns:
        Shannon Canal capacity.
    """   
    return (1/2*np.log2(1+x))

def calculateDistance(x1, y1, x2, y2):
     """
      Function for distance calculation between each point.
      Parameters:
         x1: Coordinate x of the first node.
         y1: Coordinate y of the first node.
         x2: Coordinate x of the second node.
         y2: Coordinate y of the second node.
      Returns:
        Distance between two point.
     """     
     dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist 

def FDFR(Alpha, Ps, Gsr, Gpr, Pp=10.0):
  """
      This function calculate the different parameters using Gekko.

      Parameters:
         Alpha: 1D Array containing alpha values.
         Ps: 1D Array containing Power of secondary network values.
         Gsr: 1D Array Channel Gain between secondary transmitter and relay.
      Returns:
         First SNR for Decode-and-Forward.
  """
  return (Gsr*(1-Alpha**2)*Ps**2)/(Gpr*Pp+1)

def FDF2(Alpha, Ps, Pr, Gss, Grs, Gps, Pp=10.0):
    """
      This function calculate the different parameters using Gekko.

      Parameters:
         Alpha: 1D Array containing alpha values.
         Ps: 1D Array containing Power of secondary network values.
         Pr: 1D Array containing Power of relay.
         Pp: Constant represent Power of primary network.
         Gss: 1D Array Gain between secondary transmitter and secondary receiver.
         Grs: 1D Array Gain between relay and secondary receiver.
      Returns:
         Second SNR for Decode-and-Forward.
    """
    return ((Gss*Ps**2+Grs*Pr**2)+2*(np.sqrt(Grs*Gss)*Alpha*Ps*Pr))/(Gps*Pp+1) 


def QoS_Normalized(Alpha, Pr, Ps, Grp, Gpp, Gss, Gsp):
  """
    Compute the normalized QoS.

    Parameters:
      Alpha: 1D Array containing Alpha values.
      Pr: 1D Array containing Power of relay.
      Ps: 1D Array containing Power of secondary network.
      Grp: 1D Array Channel Gain between relay and primary receiver.
      Gpp: 1D Array Channel Gain between primary transmitter and primary receiver.
      Gss: 1D Array Channel Gain between secondarytransmitter and secondary receiver.
      Gsp: 1D Array Channel Gain between secondary transmitter and primary receiver.
    Returns:
      Normalized QoS
  """
  return (((Gsp*Ps**2+Grp*Pr**2)+2*(np.sqrt(Gsp*Grp)*Alpha*Ps*Pr))-A_(Gpp))/A_(Gpp)

def mult_res(f,Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps):
  # Functions used to browse all the draws and generate Rs, Alpha, Ps, Pr for each draw
  all_res =  np.empty((0,12), float) # Empty numpy array for holding the dataset where 
  
  for i in range (len(Grp)):
    all_res = np.append(all_res, np.array(f(Grp[i], Gpp[i], Gsr[i], Gpr[i], Gss[i], Grs[i], Gsp[i], Gps[i])), axis=0)
  return all_res

def ecart(a, b):
  """
      Gap between two array.
      Parameters:
         a: First array.
         b: Second array.
      Returns:
        Gap between two array.
  """     
  return np.abs(a-b)

#------------ tensorflow functions for DNN ------------# 
def custom_sigmoid(x):
  """
    Modified sigmoid function used for handling predicted powers.

    Parameters:
      x: tensor.
    Returns:
      Output of sigmoid function range between 0 and sqrt(10)
  """
  output = tf.multiply(tf.sqrt(tf.constant(10,dtype=tf.float32)),nn.sigmoid(x))
  # Cache the logits to use for crossentropy loss.
  output._keras_logits = x  # pylint: disable=protected-access
  return output


def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2, dtype=tf.float32))
  return numerator / denominator


def stack_dtrain(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps):  # dataset_stack changed name
  """ stack different channel data used for training the network"""

  return np.stack((Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps), axis=1)

def stack_dtest(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Rs, Alpha, Pr, Ps): # changed name from dataset ==> stack_data ==>stack_dtest
  """stack different channel gains, rate, alpha and power parameters"""

  return np.stack((Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Rs, Alpha, Pr, Ps), axis=1)

def data_filter(Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps):
  s = 10**(-40/10)/10
  mask = np.all(np.stack([Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps], axis=1)>=s, axis=1)
  return Grp[mask], Gpp[mask], Gsr[mask], Gpr[mask], Gss[mask], Grs[mask], Gsp[mask], Gps[mask]

def percentage(x):
  x = np.array(x)*100
  x = x.tolist()
  return x
  
def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2, dtype=tf.float32))
  return numerator / denominator

def history_extraction(Lambda, key):
    """
        Parameters:
          Lambda : dictionary that contains the values of lambda as key (str) and values (int).
          key : str used to extract the history (loss or val_loss, throughput, val_throughput...)
        Returns:
          training history of loss or secondary_rate or primary rate degradation
    """

    
    temp, data = [], []
    
    for ld_k in Lambda.keys():
  
        history = np.load(root_dir+'/lambda = '+ld_k+'/history/'+ld_k+'.npy',allow_pickle='TRUE').item()

        temp.append(history[key])

        data.append(temp)
        temp  = []
    return data

def noise_to_channels_V1(X, channel_ID, SNRs_db = [-10, -5, 0, 5, 10, 15, 20]):
    '''
    Parameters : 
       
        test_set :  test set containing the H channels
    
        col : list of index for the specific column to add noise
    
    Returns:
    
        channel gain ndarray container of noisy channels 

    '''   
   
    var_X = np.var(X[:, channel_ID], axis=0, keepdims=True)
    
    noisy_gains = [] # list to store all the noisy H matrices with different level of noise variance
    
    for SNR_db in SNRs_db:
        
        SNR = np.power(10,SNR_db/10)
        noises = np.sqrt(var_X/SNR)*np.random.normal(0.0, 1.0, (X.shape[0], len(channel_ID)))#len(secondary_ID)
    
        X_noised = X.copy()
        X_noised[:, channel_ID] = X_noised[:, channel_ID] + noises
        noisy_gains.append(X_noised)

    return SNRs_db, np.asarray(noisy_gains, dtype="float64")



def noise_to_channels(X, primary_ID, secondary_ID, SNRs_db = [-10, -5, 0, 5, 10, 15, 20]):
    '''
    Parameters : 
       
        test_set :  test set containing the H channels
    
        col : list of index for the specific column to add noise
    
    Returns:
    
        channel gain ndarray container of noisy channels 

    '''   
   
    #static_var_X = np.sqrt(np.var(X[:, primary_ID], axis=0))

    #var_X = np.var(X[:, secondary_ID], axis=0, keepdims=True)
    var_X = np.var(X[:, secondary_ID+primary_ID], axis=0, keepdims=True)
    
    noisy_gains = [] # list to store all the noisy H matrices with different level of noise variance
    
    for SNR_db in SNRs_db:
        
        SNR = np.power(10,SNR_db/10)
        noises = np.sqrt(var_X/SNR)*np.random.normal(0.0, 1.0, (X.shape[0], len(secondary_ID+primary_ID)))#len(secondary_ID)
    
        X_noised = X.copy()
        #X_noised[:, secondary_ID] = X_noised[:, secondary_ID] + noises
        X_noised[:, secondary_ID+primary_ID] = X_noised[:, secondary_ID+primary_ID] + noises
        #X_noised[:, primary_ID] = static_var_X
        noisy_gains.append(X_noised)

    return SNRs_db, np.asarray(noisy_gains, dtype="float64")


#-------------------------------------CF----------------------------------------------------#

def AS_A_squeeze_CF(x):
    return CF_V4(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])

def generate_benchmark_CF(H_matrix): 
    '''
    bruteforce for H without noise
    '''
    with Pool() as p:
        BF_res =  p.map(AS_A_squeeze_CF, H_matrix)

    return np.squeeze(np.asarray(BF_res, dtype="float64"))


def AS_for_noisy_channels_CF(BH_matrix):
    
    '''
    Compute bruteforce method for channel gain ndarray composed of noisy channels
    '''

    BF_res = [] # list containing channels and bruteforce results (Alpha,Pr,Ps) for each noisy matrix (0, 10^-1.5, 10^-1....) 
    
    
    for i in range(BH_matrix.shape[0]) :
        X = BH_matrix[i,:,:]
        
        temp_BF_res = generate_benchmark_CF(X)
        
        BF_res.append(temp_BF_res)

        
    return np.asarray(BF_res, dtype="float64")


#----------------------------------------DF-------------------------------------------------#

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

def BF_A_squeeze_DF(x):
    return BF_A(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])

def generate_benchmark_DF(H_matrix): 
    '''
    bruteforce for H without noise
    '''
    with Pool() as p:
        BF_res =  p.map(BF_A_squeeze_DF, np.power(H_matrix, 2))

    return np.squeeze(np.asarray(BF_res, dtype="float64"))


def bruteforce_for_noisy_channels_DF(BH_matrix):
    
    '''
    Compute bruteforce method for channel gain ndarray composed of noisy channels
    '''

    BF_res = [] # list containing channels and bruteforce results (Alpha,Pr,Ps) for each noisy matrix (0, 10^-1.5, 10^-1....) 
    
    
    for i in range(BH_matrix.shape[0]) :
        X = BH_matrix[i,:,:]
        
        temp_BF_res = generate_benchmark_DF(X)
        
        BF_res.append(temp_BF_res)

        
    return np.asarray(BF_res, dtype="float64")


#-----------------------------------------------------------------------------------------#




