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
    V_0 = E+D
    d_t = (np.log(V_0/D)+((vol**2)/2)*(T-t))/(vol*np.sqrt(T-t))
    V = (E+D*norm.cdf(d_t-vol*np.sqrt(T-t)))/norm.cdf(d_t)
    while np.max(abs(V-V_0)/abs(V))>tol:
        V_0 = V
        d_t = (np.log(V_0 / D) + ((vol ** 2) / 2) * (T - t)) / (vol * np.sqrt(T - t))
        V = (E + D * norm.cdf(d_t - vol * np.sqrt(T - t))) / norm.cdf(d_t)
    return np.array(V)

def anova_2(df):
    # Use ANOVA to gauge idiosyncratic contributions
    model = ols('TOPIX_TE ~ TMC + Denso + Toyota_Industries + Aisin_Seiki + Toyota_Boshoku + Toyoda_Gosei',data=df).fit()
    anova = anova_lm(model, typ=2)
    anova_df = anova['sum_sq'].drop('Residual').to_frame()
    anova_df['TSS'] = anova_df['sum_sq'].sum()
    anova_df['k'] = anova_df['sum_sq'] / anova_df['TSS']
    return anova_df['k']

if __name__ == "__main__":
    df_ED = pd.read_excel(r'Data\df_ED.xlsx')
    E = np.array(df_ED['E_t'])
    D = np.array(df_ED['D_t'])
    print(fixed_pt_iter_BS(E, D, 0.2))
    df_ANOVA = pd.read_excel(r'Data\df_ANOVA.xlsx')
    print(anova_2(df_ANOVA))

