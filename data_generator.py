from packages import *
from parameters import *
from functions import *

# 1) changing Drawn samples 
# 2) calculateDistance function deleted, using linalg numpy function instead 
# Drawn samples from the uniform distribution which represents the position of users
# 3) changing gain_generator to channel_gain_with_gaussian_fading
# 4) changing noisy_gain_generator to noisy_channel_gain_with_gaussian_fading and code updated
# 5) gain generator to gain_model_2
# 6) Adding v_min = 10**-3, v_max = 10**3 to uniform model parameters
# 7) GPP, GPR, GRP, GSS, GSR, GRS, GSP, GPS = gain_generator(Dpp_E), gain_generator(Dpr_E), gain_generator(Drp_E), gain_generator(Dss_E), gain_generator(Dsr_E), gain_generator(Drs_E), gain_generator(Dsp_E), gain_generator(Dps_E) to g_11 = channel_gain_with_gaussian_fading(d_11, mu, sigma, alpha)
# 8) update noisy_channel_gain_with_gaussian_fading
# 9) update relay position from 0.5 to 5
# 10) update Anne model to generate h instead of g

U_P = np.random.uniform(pos_min, pos_max, (Nbr, 2))
D_P = np.random.uniform(pos_min, pos_max, (Nbr, 2))
U_S = np.random.uniform(pos_min, pos_max, (Nbr, 2))
D_S = np.random.uniform(pos_min, pos_max, (Nbr, 2))
R  = 5 * np.ones((Nbr, 2))

# Calculate distance

# Distance Primary network
d_PP = np.linalg.norm(U_P - D_P, axis=1)
d_PR = np.linalg.norm(U_P - R  , axis=1)
d_RP = np.linalg.norm(R   - D_P, axis=1)
d_PS = np.linalg.norm(U_P - D_S, axis=1)

# Distance Secondary network
d_SS = np.linalg.norm(U_S - D_S, axis=1)
d_SR = np.linalg.norm(U_S - R  , axis=1)
d_RS = np.linalg.norm(R   - D_S, axis=1)
d_SP = np.linalg.norm(U_S - D_P, axis=1)

def channel_gain_with_gaussian_fading(d, mu=0.0, sigma=7, alpha=3):  # channel gain model
  """
      channel gain model.
      Args:
         d: distance between source and destination
         mu : gaussian fading mean
         sigma: gaussian fading sigma
         alpha: is the path loss factor
      Returns:
        channel gain
  """    
  s = np.random.normal(mu, sigma, d.shape[0])
  h = s/np.sqrt(1.0 + np.power(d, alpha))
  #g = h**2
  return h

#Noise_var = 10
def noisy_channel_gain_with_gaussian_fading(d, mu=0.0, sigma=7, alpha=3, noise_var=0.1):
  """
      Added noise to the channel gain model.
      Args:
         d: distance between source and destination
         mu : gaussian fading mean
         sigma: gaussian fading sigma
         alpha: is the path loss factor
         noise_var : is the noise variance
      Returns:
        noisy channel gain
  """     
  s = np.random.normal(mu, sigma, d.shape[0])
  h = s/np.sqrt(1.0 + np.power(d, alpha))  
  h = h + np.random.normal(mu, noise_var, Nbr)  
  
  #h = h**2 # h/N_var division by the same noise variance N_var
  #h = np.sqrt(h) + np.random.normal(mu, noise_var, Nbr)  
  #h = h**2  
  return h

def gain_model_2(d): 
  """
      Channel gain model from reference [2].
      [2] Savard, Anne, and E. Veronica Belmega. "Optimal power allocation in a relay-aided cognitive network." Proceedings of the 12th EAI International Conference on Performance Evaluation Methodologies and Tools. 2019.
      
      Args:
         d: distance between source and destination
      Returns:
        channel gain
  """      
 
  return 1/d**(3/2) #(1/d**(3/2))**2

def uniform_gain(v_min = 10**-3, v_max = 10**3):
  """
      Drawn samples from the uniform distribution represents channel gain.
      Args:
         None
      Returns:
        channel gain
  """   
  # Exponential selection
  Gss =  np.random.uniform(v_min,v_max,Nbr)
  Gsr =  np.random.uniform(v_min,v_max,Nbr)
  Grs =  np.random.uniform(v_min,v_max,Nbr)
  Gpp =  np.random.uniform(v_min,v_max,Nbr)
  Gpr =  np.random.uniform(v_min,v_max,Nbr)
  Grp =  np.random.uniform(v_min,v_max,Nbr)
  Gsp =  np.random.uniform(v_min,v_max,Nbr)
  Gps =  np.random.uniform(v_min,v_max,Nbr)


  return Gpp, Gpr, Grp, Gss, Gsr, Grs, Gsp, Gps # Acces Added

def rician_fading():
    return rice.rvs(b, size=Nbr)

def nakagami_fading():
    return nakagami.rvs(nu, size=Nbr)

def gain_model_3(d):
    return np.random.normal(mu,sigma)*np.sqrt(d**-alpha)

def add_noise(x, noise_var):
    x = np.sqrt(x) + np.random.normal(mu, noise_var, x.shape[0]) 
    x = x**2
    return x

def channel_type():
    ans=True
    while ans:
        print("""
        1. Channel gain with gaussian fading [need ref]
        2. Channel gain with Uniform distribution
        3. Channel gain with Anne  model [need ref]
        4. AWGN
        5. Channel gain with Rician fading 
        6. Channel gain with Nakagami fading
        7. Noisy channel gain with gaussian fading
        8.Exit/Quit
        """)
        ans=input("Select channel gain\n")
        if ans=="1":
            g_PP = channel_gain_with_gaussian_fading(d_PP, mu, sigma, alpha)
            g_PS = channel_gain_with_gaussian_fading(d_PS, mu, sigma, alpha)
            g_PR = channel_gain_with_gaussian_fading(d_PR, mu, sigma, alpha)
            g_SP = channel_gain_with_gaussian_fading(d_SP, mu, sigma, alpha)
            g_SS = channel_gain_with_gaussian_fading(d_SS, mu, sigma, alpha)
            g_SR = channel_gain_with_gaussian_fading(d_SR, mu, sigma, alpha)
            g_RP = channel_gain_with_gaussian_fading(d_RP, mu, sigma, alpha)
            g_RS = channel_gain_with_gaussian_fading(d_RS, mu, sigma, alpha)
    
            print("Channel gain created")
            return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
            ans = None
        elif ans=="2":
          g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS  = uniform_gain()
          print("Channel gain created")
          return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
          ans = None
        elif ans =='3':
          g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS = gain_model_2(d_PP), gain_model_2(d_PR), gain_model_2(d_RP), gain_model_2(d_SS), gain_model_2(d_SR), gain_model_2(d_RS), gain_model_2(d_SP), gain_model_2(d_PS) 
          print("Channel gain created")
          return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
          ans = None
        elif ans =='4':
            g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS = gain_model_3(d_PP), gain_model_3(d_PR), gain_model_3(d_RP), gain_model_3(d_SS), gain_model_3(d_SR), gain_model_3(d_RS), gain_model_3(d_SP), gain_model_3(d_PS)
            print("Channel gain created")
            return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
            ans = None
        elif ans =='5':
            g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS = rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading(), rician_fading() 
            print("Channel gain created")
            return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
            ans = None
        elif ans =='6':
            g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS = nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading(), nakagami_fading()
            print("Channel gain created")
            return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
            ans = None
        elif ans =='7':
            #Noise_variance = float(input('Enter noise variance \n'))
            g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS = noisy_channel_gain_with_gaussian_fading(d_PP, noise_variance), noisy_channel_gain_with_gaussian_fading(d_PR, noise_variance), noisy_channel_gain_with_gaussian_fading(d_RP, noise_variance), noisy_channel_gain_with_gaussian_fading(d_SS, noise_variance), noisy_channel_gain_with_gaussian_fading(d_SR, noise_variance), noisy_channel_gain_with_gaussian_fading(d_RS, noise_variance), noisy_channel_gain_with_gaussian_fading(d_SP, noise_variance), noisy_channel_gain_with_gaussian_fading(d_PS, noise_variance) 
            print("Channel gain created")
            return g_PP, g_PR, g_RP, g_SS, g_SR, g_RS, g_SP, g_PS
            ans = None
        elif ans =='8':
          ans = None
        else:
           print("Not Valid Choice Try again")