import tool
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

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
        par = [0.05, 0.2]
        bds = [(0.001, 1), (0.001, 1)]
        result = minimize(self.likelihood, par, bounds=bds)
        return result.x

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
        mean = (miu - 0.5 * sigma ** 2) * self.dt() + np.exp(-delta*time)*np.sqrt(cro*0.5/delta)*Z_0*(1-np.exp(delta*self.dt()))
        var = (sigma ** 2) * self.dt() + (cro/(2*delta))*(1-np.exp(-2*delta*self.dt()))-((2*sigma*np.sqrt(cro))/delta)*(1-np.exp(-delta*self.dt()))*self.k_s
        sigma_shot = np.sqrt(sigma**2+cro-2*sigma*np.sqrt(cro)*self.k_s)
        V_hat = tool.fixed_pt_iter_BS(self.data[0], self.data[1], sigma_shot, T = len(self.data[0])/self.time_step)
        d_hat = (np.log(V_hat)/self.data[1]+(sigma**2+cro-2*sigma*np.sqrt(cro)*self.k_s)*self.T/2)/(np.sqrt(self.T)*np.sqrt(sigma**2+cro-2*sigma*np.sqrt(cro)*self.k_s))
        n = len(self.data[0])-1
        A = np.sum(0.5*(np.log(V_hat[1:]/V_hat[:-1])-mean)**2/var)
        B = (n/2)*np.log(var)
        C= np.sum(np.log(V_hat[1:]))
        D = np.sum(np.log(norm.cdf(d_hat[1:])))
        llh = (n/2)*np.log(2*np.pi)+A+B+C+D
        return -llh

    def optim(self):
        par = [0.03, 0.01, 0.5, 0.01, 0]
        bds = [(0.01, 0.1), (0.001, 0.3), (0.001, 5), (0.001, 0.2), (-5, 5)]
        result = minimize(self.likelihood, par, bounds=bds, method='Powell')
        return result.x

if __name__ == "__main__":
    df_ED = pd.read_excel(r'Data\df_ED.xlsx')
    E = df_ED['E_t']
    D = df_ED['D_t']
    data = np.array([E, D])
    time_step = 252
    data_len = len(E)
    obj = GBM_param(time_step = time_step , data = data)
    print(obj.optim())
    df_ANOVA = pd.read_excel(r'Data\df_ANOVA.xlsx')
    obj_2 = GBMSN_param(time_step = time_step , data = data, t = 1/time_step, T = data_len/time_step, k_s = tool.anova_2(df_ANOVA)['TMC'])
    print(obj_2.optim())