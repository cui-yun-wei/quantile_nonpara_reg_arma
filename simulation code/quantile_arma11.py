import scipy
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import numpy as np


def quantile_ARMA11_ss2(dd, SS, para_vec, tau = 0.5):
    
    
    a1  = para_vec[0]  # the AR coefficient of the first lag
    b1  = para_vec[1]  # the MA coefficient of the first lag
    
    values = dict();
    
    values['sab'] = 0
    values['alpha_tau'] = 0
    values['kappa'] = 0
    
    reg_b = np.array(para_vec[2:len(para_vec)])

    ar_root = np.roots([-a1, 1])   # AR polynomial roots
    ma_root = np.roots([b1, 1])    # MA polynomial roots
    xroot   = np.append(ar_root, ma_root)
  
    if (np.all(np.abs(xroot) > 1)):
        y  = np.array(dd)
        y_hat = np.matmul(SS, reg_b)
        y_diff = np.zeros(len(dd)+1)  # need values at time 0
        y_diff[range(1, len(y_diff))] = y - y_hat
        
        z_hat= np.zeros(len(dd)+1) 
        times = range(0, len(dd))    
        for t in times: z_hat[t+1] = y_diff[t+1] - a1*y_diff[t] - b1*z_hat[t]
        
        z_hat = np.delete(z_hat, range(1))  # delete the first extra
        sab = np.dot(tau - (z_hat < 0)*1, z_hat)
        
        values['sab'] = np.log(sab)
        values['alpha_tau']  = -np.mean(z_hat)
        values['kappa'] = -np.mean(y_diff)

    else:
        #sab = np.exp(250)
        sab = np.exp(300)
        #sab = np.exp(100*np.exp(np.min(np.abs(xroot))))
        #sab = np.exp(400*(1-np.min(np.abs(xroot))))
        values['sab'] =sab

    return values


def fit_quantile_ARMA11(dd, SS, para_0, tau = 0.5):
    
    # dd: observed time series 
    # SS: design matrix
    # para_0: initial parameter values:
    # para_0[1]: phi_1; para_0[2]: theta_1
        


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
        result = quantile_ARMA11_ss2(dd, SS, para, tau)['sab']
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
        ar_root = np.roots([-res.x[0], 1])   # AR polynomial roots
        ma_root = np.roots([res.x[1], 1])    # MA polynomial roots
        xroot   = np.append(ar_root, ma_root)

    sround = 0 

    while np.any(np.isnan(res.x)) or np.any(np.isinf(res.x)) or np.any(np.abs(xroot) <= 1):
        print("Hello, from inside quantile estimation!!!!!!")
        para_0[0:2] = np.random.uniform(size = 2)

        if sround > 25: 
            para_0[0:2] = np.random.uniform(-1, 1, size = 2)      

        try:
            res1 = minimize(les_func, para_0, method='SLSQP', options={'maxiter':2000})
        except StopOptimization as e:
            res1 = OptimizeResult()
            res1.x = e.para
            res1.fun = np.exp(300)
            res1.success = False
            res1.message = "Optimization stopped due to NaN/Inf in parameter vector."
        print(f"the target function value 1 is {res1.fun}") 
             
        bounds = [(-2, 2)]*2 + [(-5, 5)] * (len(para_0) - 2)
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
           
                
        if (not np.any(np.isnan(res.x))) and (not np.any(np.isinf(res.x))):
            ar_root = np.roots([-res.x[0], 1])   # AR polynomial roots
            ma_root = np.roots([res.x[1], 1])    # MA polynomial roots
            xroot   = np.append(ar_root, ma_root)
            print(f"Roots of the polynomial: {xroot}")
        else:
            xroot = [0.5, 0.5]
            print("Cannot compute roots: x contains NaN or Inf") 
    
        sround = sround + 1 

    return res.x
