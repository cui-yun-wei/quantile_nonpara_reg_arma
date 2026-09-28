
#least squares method for AR(1) random error implemented in Python



import scipy
from scipy.optimize import minimize
import numpy as np



def leastsq_AR1_ss(dd, SS, para_vec):
    
    a1  = para_vec[0]
      
    
    values = dict();
    
    values['sms'] = 0  # sum of squares   
   
    reg_b = np.array(para_vec[1:len(para_vec)])
    xroot = np.roots([-a1, 1])
    
    if (np.any(np.abs(xroot) <= 1)):
        sms = np.exp(300)
        values['sms'] =sms
    else:
        y  = np.array(dd)
        y  = y - np.mean(y)
        SS = SS - np.mean(SS, axis = 0)
        y_hat = np.matmul(SS, reg_b)
        
        
        y_diff = np.zeros(len(dd)+1) 
        y_diff[range(1, len(y_diff))] = y - y_hat
        
        z_hat= np.zeros(len(dd)+1) 
        times = range(0, len(dd))    
        for t in times: z_hat[t+1] = y_diff[t+1] - a1*y_diff[t]
        
        z_hat = np.delete(z_hat, range(1))  # AR(1), delete the first extra
              
                
        sms = np.dot(z_hat, z_hat)
        
        values['sms'] = np.log(sms)
   
    return values



def leastsq_AR_residuals(dd, SS, para_vec):
    
    a1  = para_vec[0]
    reg_b = np.array(para_vec[1:len(para_vec)])  
    
    y  = np.array(dd)
    y  = y - np.mean(y)
    SS = SS - np.mean(SS, axis = 0)
    y_hat = np.matmul(SS, reg_b)
        
        
    y_diff = np.zeros(len(dd)+1) 
    y_diff[range(1, len(y_diff))] = y - y_hat
        
    z_hat= np.zeros(len(dd)+1) 
    times = range(0, len(dd))    
    for t in times: z_hat[t+1] = y_diff[t+1] - a1*y_diff[t]
        
    z_hat = np.delete(z_hat, range(1))  # AR(1), delete the first extra
              
    return z_hat



def fit_leastsq_AR1(dd, SS, para_0):
    
    # dd: observed time series 
    # SS: design matrix
    # para_0: initial parameter values:
    # para_0[1]: theta_1; para_0[2]: theta_2
        
    def les_func(para):
            return leastsq_AR1_ss(dd, SS, para)['sms']    


    res = minimize(les_func, para_0,  method='SLSQP',  
                         options={'maxiter':5000, 
                                   'ftol': 1e-12, 
                                    'eps': 1e-12, 
                                     'disp': 1  })


    xroot   = np.roots([-res.x[0], 1])

    while (np.any(np.abs(xroot) <= 1)):
        
        para_0 = np.zeros(len(para_0)) 
    
        para_0[0:1] = np.random.uniform(size = 1)

        xroot   = np.roots([-para_0[0], 1])

        while (np.any(np.abs(xroot) <= 1)):
            para_0[0:1] = np.random.uniform(size = 1)
            xroot   = np.roots([-para_0[0], 1])           
    
        res = minimize(les_func, para_0,  method='SLSQP',  
                         options={'maxiter':5000, 
                                   'ftol': 1e-12, 
                                    'eps': 1e-8, 
                                     'disp': 1  })
               
        xroot   = np.roots([-res.x[0], 1]) 

    return res.x




def fit_leastsq_AR1_BFGS(dd, SS, para_0):
    
    # dd: observed time series 
    # SS: design matrix
    # para_0: initial parameter values:
    # para_0[1]: theta_1; para_0[2]: theta_2
        
    def les_func(para):
            return leastsq_AR_ss(dd, SS, para)['sms']    


    res = minimize(les_func, para_0,  method='BFGS',  
                         options={'maxiter':5000, 
                                   'gtol': 1e-9, 
                                    'eps': 1e-9, 
                                     'disp': 1})




    xroot   = np.roots([-res.x[0], 1])

    while (np.any(np.abs(xroot) <= 1)):
        
        para_0 = np.zeros(len(para_0)) 
    
        para_0[0:1] = np.random.uniform(size = 1)

        xroot   = np.roots([-para_0[0], 1])

        while (np.any(np.abs(xroot) <= 1)):
            para_0[0:1] = np.random.uniform(size = 1)
            xroot   = np.roots([-para_0[0], 1])           
    
        res = minimize(les_func, para_0,  method='BFGS',  
                         options={'maxiter':5000, 
                                   'gtol': 1e-9, 
                                    'eps': 1e-9, 
                                     'disp': 1})
               
        xroot   = np.roots([-res.x[0], 1]) 

    return res.x