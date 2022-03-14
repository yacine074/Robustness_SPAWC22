#from packages import *
#from parameters import *


import numpy as np
from scipy.stats import rice, nakagami
import scipy


def calculate_A(H11,P1=10.0,tau = 0.25): 
    return ((((H11**2)*P1)/(((1+(((H11**2)*P1)))**(1-tau))-1))-1)    

def C(x): 
    return (1/2*np.log2(1+x))


def Delta(HR1, H11, H2R, H1R, H22, HR2, H21, H12, P1 = 10.0):
   
    #A = (((H11**2*P1)/(((1+((H11**2*P1)/(N1)))**(1-tau))-1))-N1)    
    A = calculate_A(H11)
    N2_tilde = ((H12**2)*P1+1)
    NR_tilde = ((H1R**2)*P1+1)
    P_Z = ((H12*H1R*P1)/(np.sqrt(N2_tilde*NR_tilde)))
    K1 = ((H2R**2)*N2_tilde+(H22**2)*NR_tilde-2*H2R*H22*P_Z*np.sqrt(N2_tilde*NR_tilde))
    K2 = ((1-P_Z**2)*NR_tilde*N2_tilde)
    
    C1 = K1*(HR1**2)*((H22**2)*(HR1**2)-(HR2**2)*(H21**2))
    C2 = K1*(HR2**2)*(H21**2)*A-2*(HR1**2)*(H22**2)*K1*A-(HR1**2)*(H22**2)*(H21**2)*K2
    C3 = (H22**2)*K1*(A**2)+(H22**2)*(H21**2)*K2*A
    C4 = K2*(HR2**2)*(H21**2)**2-N2_tilde*K1*(H21**2)*(HR1**2)
    C5 = N2_tilde*(H21**2)*(K1*A+K2*(H21**2))
    D = C1**2*C5**2-C1*C4*(C2*C5-C3*C4)
    
    return C1,C2,C3,C4,C5,D

def f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr, P2, P1 =10.0):
    # f en fonction de PR et P2
    A = calculate_A(H11)     

    N2_tilde = ((H12**2)*P1+1)
    NR_tilde = ((H1R**2)*P1+1)
    P_Z = ((H12*H1R*P1)/(np.sqrt(N2_tilde*NR_tilde)))
    K1 = ((H2R**2)*N2_tilde+(H22**2)*NR_tilde-2*H2R*H22*P_Z*np.sqrt(N2_tilde*NR_tilde))
    K2 = ((1-P_Z**2)*NR_tilde*N2_tilde)
    #res = (K1*HR2**2*P2*Pr+H22**2*P2*K1*P2+H22**2*P2*K2)/(K2*HR2**2*Pr+N2_tilde*K1*P2+N2_tilde*K2)
    res = (K1*(HR2**2)*P2*Pr+(H22**2)*P2*(K1*P2+K2))/(K2*(HR2**2)*Pr+N2_tilde*K1*P2+N2_tilde*K2)
    
    return res

def f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x, P1 = 10.0):
    'fonction depend seulment de Pr (x apres changement de variable) '
    # f en fonction de PR 
    A = calculate_A(H11)
    N2_tilde = ((H12**2)*P1+1)
    NR_tilde = ((H1R**2)*P1+1)
    P_Z = ((H12*H1R*P1)/(np.sqrt(N2_tilde*NR_tilde)))
    K1 = ((H2R**2)*N2_tilde+(H22**2)*NR_tilde-2*H2R*H22*P_Z*np.sqrt(N2_tilde*NR_tilde))
    K2 = ((1-P_Z**2)*NR_tilde*N2_tilde)


    C1 = K1*(HR1**2)*((H22**2)*(HR1**2)-(HR2**2)*(H21**2))
    C2 = K1*(HR2**2)*(H21**2)*A-2*(HR1**2)*(H22**2)*K1*A-(HR1**2)*(H22**2)*(H21**2)*K2
    C3 = (H22**2)*K1*(A**2)+(H22**2)*(H21**2)*K2*A
    C4 = K2*(HR2**2)*(H21**2)**2-N2_tilde*K1*(H21**2)*(HR1**2)
    C5 = N2_tilde*(H21**2)*(K1*A+K2*(H21**2))

    f = (C1*x**2+C2*x+C3)/(C4*x+C5)
    
    return f
def CF_V4(HR1, H11, H2R, H1R, H22, HR2, H21, H12, P2_max=10.0, PR_max=10.0):
    '''
    Becarful of channels ID order (check the Analytic solution file channels ID order)!!!!!!
    HR2, H12, H1R, HR1, H22, H2R, H21, H11
    '''
        
    A = calculate_A(H11)
    #x_min, x_max = f_domain(HR2, H12, H1R, HR1, H22, H2R, H21, H11)
    if ( (A/(HR1**2)) < PR_max and (A/(H21**2)) < P2_max ) :  # H1

        x_min, x_max = 0.0, A/(HR1**2)        

    elif ( (A/(HR1**2)) < PR_max and (A/(H21**2)) > P2_max ):# H2

        x_min, x_max =  (A-(H21**2)*P2_max)/(HR1**2), A/(HR1**2)

    elif ( (A/(HR1**2)) > PR_max and (A/(H21**2)) < P2_max ):# H3

        x_min, x_max =  0.0 , PR_max
        
    elif ( (A/(HR1**2)) > PR_max and (A/(H21**2)) > P2_max ) and P2_max*(H21**2) + PR_max*(HR1**2) > A: #H4

        x_min, x_max =  (A-(H21**2)*P2_max)/(HR1**2), PR_max
        
    C1, C2, C3, C4, C5, delta = Delta(HR1, H11, H2R, H1R, H22, HR2, H21, H12)

    try :
        if C1*C4>0:
            #print('C1*C4>0')
            
            if delta > 0 :
                x_1 = (-(C1*C5)-np.sqrt(delta))/(C1*C4) 
                x_2 = (-(C1*C5)+np.sqrt(delta))/(C1*C4) 
                if (x_1 <= x_min <= x_max <= x_2): 
                    Pr_opt = x_min 
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt,P2_opt)
                elif (x_1<=x_min<=x_2<=x_max and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_min)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max)): 
                    Pr_opt = x_min 
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt,P2_opt)
                      
                elif x_1 >= x_max :
                    Pr_opt = x_max 
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
               
                elif x_2 <= x_max :
                    Pr_opt = x_max 
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)

                elif  (x_1 <= x_min <= x_2 <= x_max and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_min)):
                    Pr_opt = x_max 
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                    
                elif (x_min <= x_1 <= x_2<=x_max and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, H11, x_1)): 
                    Pr_opt = x_max 
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)

                elif (x_min <= x_1 <= x_max <=x_2) :
                    Pr_opt = x_1
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                elif (x_min<=x_1<=x_2<=x_max and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_1)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max)):
                    Pr_opt = x_1
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


            elif delta == 0 :
                
                x_0 = -C5/C4
                Pr_opt = x_max
                P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                
            else :
                    Pr_opt = x_max
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                    


        elif C1*C4 == 0.0 : 
            #print('C1*C4 == 0')
            x_ = -(C2*C5-C3*C4)/(2*C1*C5)

            if (C1*C5)>0.0 :
                
                if (C2*C5-C3*C4)>0.0 :

                    Pr_opt = x_max
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)

                else : 

                    if x_ < x_min : 

                        Pr_opt = x_max
                        P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                        SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)

                    elif (x_min <= x_ <= x_max) : 

                        if f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_min)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max) :
                            
                            Pr_opt = x_min
                            P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                            SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


                        else : 
                            
                            Pr_opt = x_max
                            P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                            SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


                    elif x_ > x_max :

                        Pr_opt = x_min
                        P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                        SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


            elif (C1*C5) == 0.0 : 
                #print('C1*C5')
                if (C2*C5-C3*C4) > 0.0 :
                    
                    Pr_opt = x_max
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)

                else :
                    
                    Pr_opt = x_min
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


            else :
                if (C2*C5-C3*C4)>0.0 :
                    #print((C2*C5-C3*C4))
                    if x_ < x_min :
                        
                        Pr_opt = x_min
                        P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                        SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


                    elif (x_min <= x_ <= x_max):
                        
                        Pr_opt = x_
                        P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                        SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)

                    elif x_ > x_max :
                        
                        Pr_opt = x_max  #     x_    
                        P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                        SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                
                else : # (C2*C5-C3*C4) < 0 ...
                        Pr_opt = x_min        
                        P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                        SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


        else :
            #print('C1*C4<0')

            if delta > 0.0 :
                #print('delta>0')
                x_1 = (-(C1*C5)-np.sqrt(delta))/(C1*C4) 
                x_2 = (-(C1*C5)+np.sqrt(delta))/(C1*C4)
                
                if x_1 <= x_min  : 
                    
                    Pr_opt = x_min
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                
                elif (x_min<=x_2<=x_max and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_min)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12,x_max)): 
                    Pr_opt = x_min
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


                elif  x_max <= x_2:
                    Pr_opt = x_min
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                elif  (x_min<=x_2<=x_max<=x_1 and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12,x_min)):
                    Pr_opt = x_min
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)


                elif (x_min <= x_1 <= x_max) :
                    
                    Pr_opt = x_1
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                elif (x_min <= x_2 <= x_max <= x_1 and f(HR1, H11, H2R, H1R, H22, HR2, H21, H12, x_max)>f(HR1, H11, H2R, H1R, H22, HR2, H21, H12,x_min)) :
                    
                    Pr_opt = x_max
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                elif (x_2 <= x_min <= x_max <= x_1) :
                    
                    Pr_opt = x_max
                    P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                    SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
               
                
            elif delta == 0 :
                #print('delta=0')


                x_0 = -C5/C4
                Pr_opt = x_min
                P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                
            else :
            
                Pr_opt = x_min
                P2_opt = (A-(HR1**2)*Pr_opt)/(H21**2)
                SNR = f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt)
                

     

    except UnboundLocalError:
        
        Pr_opt, P2_opt, SNR = PR_max, P2_max ,f_obj(HR1, H11, H2R, H1R, H22, HR2, H21, H12, PR_max, P2_max)
        #print('[H5]-[H6]')
    
    
    return HR1, H11, H2R, H1R, H22, HR2, H21, H12, Pr_opt, P2_opt, SNR
            
    
    
    