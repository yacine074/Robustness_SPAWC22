from packages import *
# changing p_min, p_max to pos_min, pos_max
# changing Pp to Pp_max
# changing pathloss_factor variable to alpha

#initialization of parameters
Pp = 10.0 # primary network maximum primary power
Pr_max, Ps_max = 10.0, 10.0 # maximum relay power, secondary network power

Alpha_tilde = math.sqrt(1)
Psmax_tilde = math.sqrt(Ps_max)
Prmax_tilde = math.sqrt(Pr_max)

alpha = 3.0 # is the path loss factor

tau = 0.25

Nbr = int(2E6) # Number of configuration

pos_min = 0 # min position in cell
pos_max = 10 # max position in cell

v_min = 10**-3 # min channel value gain for uniform distribution
v_max = 10**3  # max channel value gain for uniform distribution 

mu, sigma = 0, 7 # mean and standard deviation for channel_gain_with_gaussian_fading

QoS_thresh = -5 # QoS_threshold

b = 0.775 # rice
nu = 4.97 # nakagami

noise_variance = 0.1

root_dir ='DNN'
BASE_PATH = "/path/to/base"#/path/to/base
