#least squares method for MA(2) random error implemented in Python




import scipy
from scipy.optimize import minimize
import numpy as np



def leastsq_MA_ss(dd, SS, para_vec):
    
    tau = 0.5
    b1  = para_vec[0]
    b2  = para_vec[1]  # it should be the coefficient of the second lag
    
    values = dict();
    
    values['sms'] =0  # sum of squares
   
    
    reg_b = np.array(para_vec[2:len(para_vec)])
    
     
    xroot = np.roots([b2, b1, 1])
    
    if (np.any(np.abs(xroot) < 1)):
        sms = np.exp(300)
        values['sms'] =sms
    else:
        y  = np.array(dd)
        
        y  = y - np.mean(y)
        
        SS = SS - np.mean(SS, axis = 0)
        
        y_hat = np.matmul(SS, reg_b)
        
        y_diff= y - y_hat
        
        z_hat= np.zeros(len(dd)+2) 
        times = range(0, len(dd))    
        for t in times: z_hat[t+2] = y_diff[t] - b1*z_hat[t+1] - b2*z_hat[t]
        
        z_hat = np.delete(z_hat, range(2))  # MA(2), delete the first two extra
              
                
        sms = np.dot(z_hat, z_hat)
        
        values['sms'] = np.log(sms)
   
    return values


def fit_leastsq_MA2(dd, SS, para_0):
    
    # dd: observed time series 
    # SS: design matrix
    # para_0: initial parameter values:
    # para_0[1]: theta_1; para_0[2]: theta_2
        
    def les_func(para):
            return leastsq_MA_ss(dd, SS, para)['sms']    

    res = minimize(les_func, para_0,  method='SLSQP',  
                    options={'maxiter':5000, 'ftol' : 1e-5})


    
    xroot   = np.roots([res.x[1],res.x[0], 1])

    while (np.any(np.abs(xroot) <= 1)):
        
        para_0 = np.zeros(len(para_0)) 
    
        para_0[0:2] = np.random.uniform(size = 2)

        
        xroot   = np.roots([para_0[1], para_0[0], 1])

        while (np.any(np.abs(xroot) <= 1)):
            para_0[0:2] = np.random.uniform(size = 2)
            xroot   = np.roots([para_0[1], para_0[0], 1])             
    
        res = minimize(les_func, para_0,  method='SLSQP',  
                    options={'maxiter':1000, 'ftol' : 1e-5})
               
        xroot   = np.roots([res.x[1],res.x[0], 1]) 


    return res.x


