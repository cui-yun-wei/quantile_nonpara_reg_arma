import scipy
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import numpy as np


def quantile_MA1_ss2(dd, SS, para_vec, tau = 0.5):
    
    
    b1  = para_vec[0]
        
    values = dict();
    
    values['sab'] =0
    values['alpha_tau']  = 0
    values['kappa'] = 0
       
    
    reg_b = np.array(para_vec[1:len(para_vec)])
    
     
    xroot = np.roots([b1, 1])
    
    if (np.any(np.abs(xroot) < 1)):
        sab = np.exp(300)
        values['sab'] =sab
    else:
        y  = np.array(dd)
        y_hat = np.matmul(SS, reg_b)
        
        y_diff= y - y_hat
        
        z_hat= np.zeros(len(dd)+1) 
        times = range(0, len(dd))    
        for t in times: z_hat[t+1] = y_diff[t] - b1*z_hat[t]
        

        z_hat = np.delete(z_hat, range(1))  # MA(1), delete the first extra
              
        sab = np.dot(tau - (z_hat < 0)*1, z_hat)
        
        values['sab'] = np.log(sab)
        values['alpha_tau']  = -np.mean(z_hat)
        values['kappa'] = -np.mean(y_diff)
    
    return values


def fit_quantile_MA1(dd, SS, para_0, tau = 0.5):
    
    # dd: observed time series 
    # SS: design matrix
    # para_0: initial parameter values:
    # para_0[1]: theta_1

    # Define custom exception inside the function
    class StopOptimization(Exception):
        def __init__(self, para):
            super().__init__("NaN or Inf detected in parameter vector.")
            self.para = para

    # Define the objective function
    def les_func(para):
        if np.any(np.isnan(para)) or np.any(np.isinf(para)):
            print("NaN or Inf detected in parameter vector:", para)
            raise StopOptimization(para)
        
        # Replace with your real computation
        result = quantile_MA1_ss2(dd, SS, para, tau)['sab']
        return result


    try:
        res= minimize(les_func, para_0,  method='SLSQP', options={'maxiter':2000}) 
    except StopOptimization as e:
        res = OptimizeResult()
        res.x = e.para
        res.fun = np.exp(300)
        res.success = False
        res.message = "Optimization stopped due to NaN/Inf in parameter vector."
      
    if not (np.any(np.isnan(res.x)) or np.any(np.isinf(res.x))):
        xroot = np.roots([res.x[0], 1])

    while np.any(np.isnan(res.x)) or np.any(np.isinf(res.x)) or np.any(np.abs(xroot) <= 1):
        print("Hello, from inside quantile!!!!!!")
        para_0[0:1] = np.random.uniform(size = 1)

        xroot   = np.roots([para_0[0], 1])

        try:
            res1 = minimize(les_func, para_0, method='SLSQP', options={'maxiter':2000})
        except StopOptimization as e:
            res1 = OptimizeResult()
            res1.x = e.para
            res1.fun = np.exp(300)
            res1.success = False
            res1.message = "Optimization stopped due to NaN/Inf in parameter vector."
        print(f"the target function value 1 is {res1.fun}") 
             
        bounds = [(-2, 2)] + [(-5, 5)] * (len(para_0) - 1)
        try:
            res2 = dual_annealing(les_func, bounds=bounds, maxiter=1000) 
        except StopOptimization as e:
            res2 = OptimizeResult()
            res2.x = e.para
            res2.fun = np.exp(300)
            res2.success = False
            res2.message = "Optimization stopped due to NaN/Inf in parameter vector."
        print(f"the target function value 2 is {res2.fun}")
            
        res = res2 if res2.fun <= res1.fun else res1   
           
                
        if not np.any(np.isnan(res.x)) and not np.any(np.isinf(res.x)):
            xroot = np.roots([res.x[0], 1])
            print(f"Roots of the polynomial: {xroot}")
        else:
            xroot = [0.5, 0.5]
            print("Cannot compute roots: x contains NaN or Inf")


    return res.x

