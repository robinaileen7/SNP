import tool
import optim
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

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
        par = [0.05, 0.2]
        # bds = [(0.001, 1), (0.001, 1)]
        bds = np.array([[0.001, 1], [0.001, 1]])
        # result = minimize(self.likelihood, par, bounds=bds)
        result = optim.optim_non_drvt(x_0=par, x_range=bds, my_function=self.likelihood, mat='NA').min_line_search(x_0=par, tol_Brent=0.001, tol_gr=0.001)
        # result = optim.optim_non_drvt(x_0=par, x_range=bds, my_function=self.likelihood).mod_Powell()
        # return result.x
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
        # par = [0.03, 0.01, 0.5, 0.01, 0]
        # par = [0.02, 0.2, 2, 0.5, 0]
        # bds = [(0.01, 0.1), (0.001, 0.3), (0.001, 5), (0.001, 0.2), (-5, 5)]
        bds = np.array([[0.001, 0.05], [0.001, 0.3], [0.001, 10], [0.001, 1], [-10, 10]])
        par = np.mean(bds,axis=1)
        n_vector = 10
        e_list_list = []
        for j in range(len(par)):
            incr = (par[j] - bds[j][0]) / n_vector
            e_list = []
            for i in range(1, n_vector + 1):
                e_list.append(i * incr)
                e_list.append(-i * incr)
            e_list_list.append(e_list)
        e_input = np.array([[x[i] for x in e_list_list] for i in range(len(e_list_list[0]))])
        # result = minimize(self.likelihood, par, bounds=bds, method='Powell')
        # result = optim.optim_non_drvt(x_0=par, x_range=bds, my_function=self.likelihood).min_line_search(x_0=x_0)
        result = optim.optim_non_drvt(x_0=par, x_range=bds, my_function=self.likelihood, mat=e_input).mod_Powell()
        #return result.x
        return result[0]

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