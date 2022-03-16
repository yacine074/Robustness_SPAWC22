

#from packages import *
#from parameters import *
from functions import *
import tensorflow as tf



def loss_DF_WN(Lambda=10**0.5, v_tau=0.25):
    def DF_loss(G, y_out):
        ''' compute loss without normalization'''
        
        Tau = tf.constant(v_tau, dtype=tf.float32) # ==> Tau 

        W = tf.constant(Lambda, dtype=tf.float32)  # ==> lambda 

        G = tf.cast(G, dtype='float32')
        y_out = tf.cast(y_out, dtype='float32')

        # index retrieval

        Grp_indx, Gpp_indx, Gsr_indx, Gpr_indx, Gss_indx, Grs_indx, Gsp_indx, Gps_indx  = [0], [1], [2], [3], [4], [5], [6], [7]
        Alpha_indx, Pr_indx, Ps_indx  = [0], [1], [2]

        # tensors retrieval
        Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Alpha, Pr, Ps = tf.gather(G, Grp_indx, axis=1), tf.gather(G, Gpp_indx, axis=1), tf.gather(G, Gsr_indx, axis=1), tf.gather(G, Gpr_indx, axis=1), tf.gather(G, Gss_indx, axis=1), tf.gather(G, Grs_indx, axis=1), tf.gather(G, Gsp_indx, axis=1), tf.gather(G, Gps_indx, axis=1), tf.gather(y_out, Alpha_indx, axis=1), tf.gather(y_out, Pr_indx, axis=1), tf.gather(y_out, Ps_indx, axis=1)

        #  Primary power Creation

        Pp = tf.multiply(tf.ones(tf.shape(Pr), dtype=tf.dtypes.float32),10)

        # SNR1 : (Gsr*(1-alpha**2)*Ps**2)/(Gpr*Pp+1)
        SNR1 = tf.multiply(Gsr,(tf.multiply(tf.subtract(tf.constant(1,dtype=tf.float32), tf.pow(Alpha, 2)), tf.pow(Ps, 2))))
        SNR1 = tf.divide(SNR1, tf.add(tf.multiply(Gpr, Pp), tf.constant(1,dtype=tf.float32)))

        # SNR2 : ((Gss*Ps**2+Grs*Pr**2)+2*(np.sqrt(Grs*Gss)*Alpha*Ps*Pr)) ==> L1+L2/Gps*Pp+1
        L1 = tf.add(tf.multiply(Gss,tf.pow(Ps,2)),tf.multiply(Grs,tf.pow(Pr,2)))
        L2 = tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.multiply(tf.sqrt(tf.multiply(Grs,Gss)),Ps),tf.multiply(Alpha,Pr)))

        SNR2 = tf.add(L1,L2)
        SNR2= tf.divide(SNR2, tf.add(tf.multiply(Gps, Pp),tf.constant(1,dtype=tf.float32)))

        SNR_opt = tf.minimum(SNR1, SNR2)

        ########### QoS ################

        # function A' ==> A'(Gpp) : ((Gpp*Pp)/((1+(Gpp*Pp))**(1-tau)-1))-1 ==> (Gpp*Pp)/(R1) 
        R1 = tf.add(tf.constant(1, dtype=tf.float32),tf.multiply(Gpp,Pp))
        R1 = tf.pow(R1, tf.math.subtract(tf.constant(1, dtype=tf.float32),Tau))
        R1 = tf.math.subtract(R1,tf.constant(1, dtype=tf.float32))

        A_ = tf.subtract(tf.divide(tf.multiply(Gpp,Pp),R1),tf.constant(1, dtype=tf.float32))


        #Qos = (Gsp*Ps**2+Grp*Pr**2+2*np.sqrt(Gsp*Grp)*Alpha*Ps*Pr)-A_
        Qos = tf.add(tf.add(tf.multiply(Gsp,tf.pow(Ps,2)),tf.multiply(Grp,tf.pow(Pr,2))),tf.multiply(tf.constant(2,dtype=tf.float32),tf.multiply(tf.sqrt(tf.multiply(Gsp,Grp)),tf.multiply(Ps,tf.multiply(Alpha,Pr)))))

        Qos = tf.subtract(Qos, A_)

        n_Qos = tf.multiply(W,tf.keras.activations.relu(Qos)) 

        Rs_opt =  tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_opt)))

        #-n_SNR+n_Qos
        res = tf.reduce_mean(-Rs_opt+n_Qos) 
        return res
    return DF_loss


def loss_CF(Lambda=10**0.5, v_tau=0.25):
    
    def CF_loss(G, y_out):
        
        ''' compute loss for CF Relaying'''
        
        Tau = tf.constant(v_tau, dtype=tf.float32) # ==> Tau 

        W = tf.constant(Lambda, dtype=tf.float32)  # ==> lambda 

        G = tf.cast(G, dtype='float32')
        
        y_out = tf.cast(y_out, dtype='float32')

        # index retrieval

        Grp_indx, Gpp_indx, Gsr_indx, Gpr_indx, Gss_indx, Grs_indx, Gsp_indx, Gps_indx  = [0], [1], [2], [3], [4], [5], [6], [7]
        
        Pr_indx, Ps_indx  = [0], [1]

        # tensors retrieval
        
        Grp, Gpp, Gsr, Gpr, Gss, Grs, Gsp, Gps, Pr, Ps = tf.gather(G, Grp_indx, axis=1), tf.gather(G, Gpp_indx, axis=1), tf.gather(G, Gsr_indx, axis=1), tf.gather(G, Gpr_indx, axis=1), tf.gather(G, Gss_indx, axis=1), tf.gather(G, Grs_indx, axis=1), tf.gather(G, Gsp_indx, axis=1), tf.gather(G, Gps_indx, axis=1), tf.gather(y_out, Pr_indx, axis=1), tf.gather(y_out, Ps_indx, axis=1)

        #  Primary power Creation

        Pp = tf.multiply(tf.ones(tf.shape(Pr), dtype=tf.dtypes.float32),10.0)

        # NS_Tilde :NS = gPS*PP +1

        NS_Tilde = tf.add(tf.multiply(Gps,Pp),tf.constant(1, dtype=tf.float32))

        #NR_Tilde : gPR*PP +1

        NR_Tilde = tf.add(tf.multiply(Gpr,Pp),tf.constant(1, dtype=tf.float32))

        # Rho_Z : sqrt(Gpr*Gps)*Pp/sqrt(NR_tilde*NS_tilde)

        Rho_Z = tf.divide(tf.multiply(tf.sqrt(tf.multiply(Gpr,Gps)),Pp),tf.sqrt(tf.multiply(NR_Tilde,NS_Tilde)))

        #K1 : Gsr*NS_Tilde+Gss*NR_Tilde-2*Rho_Z*sqrt(Gsr*Gss*NR_Tilde*NS_Tilde)

        E1 = tf.add(tf.multiply(Gsr,NS_Tilde),tf.multiply(Gss,NR_Tilde))

        E2 = tf.sqrt(tf.multiply(tf.multiply(tf.multiply(Gsr,Gss),NR_Tilde),NS_Tilde))

        K1 = tf.math.subtract(E1,tf.multiply(tf.multiply(tf.constant(2, dtype=tf.float32),Rho_Z),E2))

        #K2 : (1-Rho_Z**2)*NR_Tilde*NS_Tilde
        
        K2 = tf.multiply(tf.math.subtract(tf.constant(1, dtype=tf.float32),tf.pow(Rho_Z,tf.constant(2, dtype=tf.float32))),tf.multiply(NR_Tilde,NS_Tilde))

        # function A' ==> A'(Gpp) : ((Gpp*Pp)/((1+(Gpp*Pp))**(1-tau)-1))-1 ==> (Gpp*Pp)/(R1) 

        R1 = tf.add(tf.constant(1, dtype=tf.float32),tf.multiply(Gpp,Pp))
        R1 = tf.pow(R1, tf.math.subtract(tf.constant(1, dtype=tf.float32),Tau))
        R1 = tf.math.subtract(R1,tf.constant(1, dtype=tf.float32))

        A_ = tf.subtract(tf.divide(tf.multiply(Gpp,Pp),R1),tf.constant(1, dtype=tf.float32))

        # QoS : 

        # PR**2 and PS**2 because custom_sigmoid(x):  
        # returns : Output of sigmoid function range between 0 and sqrt(10)

        Qos = tf.add(tf.multiply(Gsp,tf.pow(Ps,2)),tf.multiply(Grp,tf.pow(Pr,2)))

        Qos = tf.subtract(Qos, A_)

        Qos = tf.multiply(W,tf.keras.activations.relu(Qos)) 

        # SNR : 
        num = tf.add(tf.multiply(tf.multiply(tf.multiply(K1,Grs),\
                                             tf.pow(Ps,2)),tf.pow(Pr,2)),\
                     tf.add(tf.multiply(tf.multiply(Gss,tf.pow(Ps,2)),tf.multiply(K1,tf.pow(Ps,2))),\
                     tf.multiply(tf.multiply(Gss,tf.pow(Ps,2)),K2)))
        
        d_num = tf.add(tf.multiply(tf.multiply(K2,Grs), tf.pow(Pr,2)),\
                       tf.add(tf.multiply(NS_Tilde,tf.multiply(K1,tf.pow(Ps,2))),\
                       tf.multiply(NS_Tilde, K2)))
        
        SNR_opt = tf.divide(num, d_num)

        # R_S

        Rs_opt =  tf.multiply(tf.constant(0.5, dtype=tf.float32),log2(tf.add(tf.constant(1,dtype=tf.float32),SNR_opt)))

        #-n_SNR+n_Qos
        res = tf.reduce_mean(-Rs_opt+Qos) 


        return res
    return CF_loss




