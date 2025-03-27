import tool
import optim
import numpy as np
from scipy.stats import norm

def safe_sqrt(x):
    if x >=0:
        return np.sqrt(x)
    else:
        return 0

class GBM_param:
    def __init__(self, time_step, data):
        self.time_step = time_step
        self.data = data

    def dt(self):
        return 1 / self.time_step

    def likelihood(self, par):
        miu, sigma = par
        mean = (miu - 0.5 * sigma ** 2) * self.dt()
        var = (sigma ** 2) * self.dt()
        # Log-likelihood by Duan
        V_hat = tool.fixed_pt_iter_BS(self.data[0], self.data[1], sigma, T = len(self.data[0])/self.time_step)
        A = len(self.data[0])/2
        B = np.log(2*np.pi)
        C= V_hat[1:]
        D = V_hat[:-1]
        llh = -A*B-A*np.log(var)-np.sum(np.log(C))-np.sum((1/(2*var))*(np.log(C/D)-mean)**2)
        return -llh

    def optim(self):
        bds = np.array([[0.001, 1], [0.001, 1]])
        par = np.mean(bds,axis=1)
        result = optim.optim_non_drvt(x_0=par, x_range=bds, my_function=self.likelihood).mod_Powell()
        print(result[0])
        return result[0]

class GBMSN_param:
    def __init__(self, time_step, data, t, T, k_s):
        self.time_step = time_step
        self.data = data
        self.t = t
        self.T = T
        self.k_s = k_s

    def dt(self):
        return 1 / self.time_step

    def likelihood(self, par):
        miu, sigma, delta, cro, Z_0 = par
        time = np.arange(self.dt(), self.T-self.t, self.dt())
        # Mean and Var for GBM + SN computed by Egami and Kevkhishvili
        mean = (miu - 0.5 * sigma ** 2) * self.dt() + np.exp(-delta*time)*safe_sqrt(cro*0.5/delta)*Z_0*(1-np.exp(delta*self.dt()))
        var = (sigma ** 2) * self.dt() + (cro/(2*delta))*(1-np.exp(-2*delta*self.dt()))-((2*sigma*safe_sqrt(cro))/delta)*(1-np.exp(-delta*self.dt()))*self.k_s
        sigma_shot = np.sqrt(sigma**2+cro-2*sigma*safe_sqrt(cro)*self.k_s)
        V_hat = tool.fixed_pt_iter_BS(self.data[0], self.data[1], sigma_shot, T = len(self.data[0])/self.time_step)
        d_hat = ((np.log(V_hat)/self.data[1]+(sigma**2+cro-2*sigma*safe_sqrt(cro)*self.k_s))*self.T/2)/(np.sqrt(self.T)*np.sqrt(sigma**2+cro-2*sigma*safe_sqrt(cro)*self.k_s))
        n = len(self.data[0])-1
        A = np.sum(0.5*(np.log(V_hat[1:]/V_hat[:-1])-mean)**2/var)
        B = (n/2)*np.log(var)
        C= np.sum(np.log(V_hat[1:]))
        D = np.sum(np.log(norm.cdf(d_hat[1:])))
        llh = (n/2)*np.log(2*np.pi)+A+B+C+D
        return llh

    def optim(self):
        bds = np.array([[0.01, 0.1], [0.001, 0.5], [0.001, 100], [0.001, 100], [-10, 10]])
        par = np.mean(bds,axis=1)
        result = optim.optim_non_drvt(x_0=par, x_range=bds, my_function=self.likelihood).mod_Powell()
        return result[0]

if __name__ == "__main__":
    import config
    time_step = 252

    data = config.data
    est_T = config.data_len
    df_ANOVA = config.df_ANOVA
    Company = config.company

    obj = GBM_param(time_step = time_step , data = data)
    print(obj.optim())

    obj_SN = GBMSN_param(time_step = time_step , data = data, t = 1/time_step, T = est_T/time_step, k_s = tool.anova_2(df_ANOVA)[Company])
    print(obj_SN.optim())