import numpy as np
import tool
import param
import matplotlib.pyplot as plt
import config

class GBM_asset_sim:
    def __init__(self, miu, sigma, miu_1, sigma_1, delta, cro, Z_0, S_0, T, t, time_step, N):
        self.miu = miu
        self.sigma = sigma
        self.miu_1 = miu_1
        self.sigma_1 = sigma_1
        self.delta = delta
        self.cro = cro
        self.Z_0 = Z_0
        self.S_0 = S_0
        self.T = T
        self.t = t
        self.time_step = time_step
        self.N = N

    def dt(self):
        return 1 / self.time_step

    def n_path(self):
        return round((self.T - self.t) * self.time_step) + 1

    def simple_GBM(self, seed):
        # Simulate GBM process
        n = self.n_path()
        S_array = np.zeros((self.N, n))
        # Take Log to Speed Up the Simulation
        S_array[:, 0] = np.log(self.S_0)
        np.random.seed(seed)
        W = np.random.normal(0, 1, (self.N, n))
        for i in range(1, n):
            S_array[:, i] = S_array[:, i - 1] + (self.miu - 0.5 * self.sigma**2) * self.dt() + W[:, i-1] * np.sqrt(self.dt())
        ret = np.exp(S_array).mean(axis = 0)
        return ret

    def shot_GBM(self, seed):
        # Simulate Shot-Noise Process
        n = self.n_path()
        S_array = np.zeros((self.N, n))
        # Take Log to Speed Up the Simulation
        S_array[:, 0] = np.log(self.S_0)
        OU_array = np.zeros((self.N, n))
        Z_array = np.zeros((self.N, n))
        S_shot_array = np.zeros((self.N, n))
        S_shot_array[:, 0] = np.log(self.S_0)
        np.random.seed(seed)
        W = np.random.normal(0, 1, (self.N, n))
        W_shot = np.random.normal(0, 1, (self.N, n))
        for i in range(1, n):
            S_array[:, i] = S_array[:, i - 1] + (self.miu_1 - 0.5 * self.sigma_1 ** 2) * self.dt() + W[:, i - 1] * np.sqrt(self.dt())
            OU_array[:, i] = (1-self.delta*self.dt())*OU_array[:, i-1]+W_shot[:, i-1]*self.dt() ** (1 / 2)
            Z_array[:, i] = self.Z_0*np.exp(-self.delta*self.dt())+np.sqrt(2*self.delta)*OU_array[:, i]-self.Z_0
            S_shot_array[:, i] = S_array[:,i]+Z_array[:,i]*np.sqrt(self.cro/(2*self.delta))
        ret = np.exp(S_shot_array).mean(axis = 0)
        return ret


if __name__ == "__main__":
    seed = 0
    time_step = 252
    N = 10000
    S_0 = 140
    sim_T = 5
    sim_t = 0

    data = config.data
    est_T = config.data_len
    df_ANOVA = config.df_ANOVA
    Company = config.company

    obj_GBM = param.GBM_param(time_step = time_step, data = data)
    GBM_param = obj_GBM.optim()

    obj_GBMSN = param.GBMSN_param(time_step = time_step , data = data, t = 1/time_step, T = est_T/time_step, k_s = tool.anova_2(df_ANOVA)[Company])
    SN_param = obj_GBMSN.optim()

    obj_sim = GBM_asset_sim(miu=GBM_param[0], sigma=GBM_param[1], miu_1=SN_param[0], sigma_1=SN_param[1], delta=SN_param[2], cro=SN_param[3], Z_0=SN_param[4], S_0=S_0, T=sim_T, t=sim_t, time_step=time_step, N=N)

    #obj_sim = GBM_asset_sim(miu=0.0268313126894394, sigma=0.07666575043370744, miu_1=0.02670044984862588, sigma_1=0.09540626142764669,
                            #delta=0.001305145263671875, cro=0.2, Z_0=-9, S_0=S_0, T=sim_T, t=sim_t,
                            #time_step=time_step, N=N)
    simple_path = obj_sim.simple_GBM(seed)
    shot_path = obj_sim.shot_GBM(seed)

    x = np.arange(0, int(time_step*sim_T) + 1, 1)
    plt.plot(x, simple_path, label='GBM')
    plt.plot(x, shot_path, label='GBM+SN')
    plt.legend()
    plt.show()