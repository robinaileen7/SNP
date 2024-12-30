import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
def fixed_pt_iter_BS(E, D, vol):
    # Fixed-point iteration to get asset value from equity and debt
    tol = 0.00001
    T = 1
    t = 0
    V_0 = np.ones((1, len(E)))*10000
    d_t = (np.log(V_0/D)+((vol**2)/2)*(T-t))/(vol*np.sqrt(T-t))
    V = (E+D*norm.cdf(d_t-vol*np.sqrt(T-t)))/norm.cdf(d_t)
    while np.max(abs(V-V_0)/abs(V))>tol:
        V_0 = V
        d_t = (np.log(V_0 / D) + ((vol ** 2) / 2) * (T - t)) / (vol * np.sqrt(T - t))
        V = (E + D * norm.cdf(d_t - vol * np.sqrt(T - t))) / norm.cdf(d_t)
    return V

def anova_2(df):
    model = ols('y ~ x1 + x2',data=df).fit()
    print(model.params)
    anova = anova_lm(model, typ=2)
    return anova

if __name__ == "__main__":
    E = [100, 120, 125, 140]
    D = [95, 110, 120, 125]
    print(fixed_pt_iter_BS(E, D, 0.2))
    df = pd.DataFrame({'y': [11, 15, 17, 17, 20],
                       'x1': [1, 3, 4, 6, 6],
                       'x2': [1, 2, 3, 5, 7]
                       })
    print(anova_2(df))

