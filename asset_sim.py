import numpy as np

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

    def simple_GBM(self):
        # Simulate GBM process
        n = self.n_path()
        S_array = np.zeros((self.N, n))
        # Take Log to Speed Up the Simulation
        S_array[:, 0] = np.log(self.S_0)
        np.random.seed(10)
        W = np.random.normal(0, 1, (self.N, n))
        for i in range(1, n):
            S_array[:, i] = S_array[:, i - 1] + (self.miu - 0.5 * self.sigma**2) * self.dt() + W[:, i-1] * np.sqrt(self.dt())
        return np.exp(S_array).mean(axis = 0)

    def shot_GBM(self):
        # Simulate Shot-Noise Process
        n = self.n_path()
        S_array = np.zeros((self.N, n))
        # Take Log to Speed Up the Simulation
        S_array[:, 0] = np.log(self.S_0)
        OU_array = np.zeros((self.N, n))
        Z_array = np.zeros((self.N, n))
        S_shot_array = np.zeros((self.N, n))
        np.random.seed(10)
        W = np.random.normal(0, 1, (self.N, n))
        W_shot = np.random.normal(0, 1, (self.N, n))
        for i in range(1, n):
            S_array[:, i] = S_array[:, i - 1] + (self.miu_1 - 0.5 * self.sigma_1 ** 2) * self.dt() + W[:, i - 1] * np.sqrt(self.dt())
            OU_array[:, i] = (1-self.delta*self.dt())*OU_array[:, i-1]+W_shot[:, i-1]*self.dt() ** (1 / 2)
            Z_array[:, i] = self.Z_0*np.exp(-self.delta*self.dt())+np.sqrt(2*self.delta)*OU_array[:, i]-self.Z_0
            S_shot_array[:, i] = S_array[:,i]+Z_array[:,i]*np.sqrt(self.cro/(2*self.delta))
        return np.exp(S_shot_array).mean(axis = 0)


if __name__ == "__main__":
    time_step = 252
    N = 10000
    obj = GBM_asset_sim(miu=0.05, sigma=0.2, miu_1=0.05, sigma_1=0.2, delta=5, cro=0.003, Z_0=-10, S_0=100, T=1, t=0, time_step=time_step, N=N)
    simple_path = obj.simple_GBM()
    shot_path = obj.shot_GBM()
    print(simple_path[-1])
    print(shot_path[-1])