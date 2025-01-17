import numpy as np
from scipy.optimize import minimize

class optim_non_drvt:
    # This class collects optimization methods when f(x) doesn't have a derivative

    def __init__(self, x_0, x_range, my_function):
        self.x_0 = x_0
        self.x_range = x_range
        self.n = len(x_0)
        self.my_function = my_function

    def golden_ratio(self, a_b, x, which_var):
        ratio_1 = 0.618
        ratio_2 = 1 - ratio_1
        iter_gr = 10000
        tol_gr = 0.001
        ax = a_b[0]
        bx = a_b[1]
        a = min(ax, bx)
        b = max(ax, bx)
        # Pick x_0 and x_1 within [a,b] according to the Golden ratio
        x_0 = a + ratio_2*(b-a)
        x_1 = a + ratio_1*(b-a)
        x_init_0 = x.copy()
        x_init_1 = x.copy()
        x_init_0[which_var] = x_0
        x_init_1[which_var] = x_1
        f_0 = self.my_function(x_init_0)
        f_1 = self.my_function(x_init_1)
        # If f(x_0) < f(x_1) => search within [a, x_1) and else (x_0, b]
        for i in range(iter_gr):
            a = a if f_0<=f_1 else x_0
            b = x_1 if f_0<=f_1 else b
            x_0 = a + ratio_2*(b-a) if f_0<=f_1 else x_1
            x_1 = x_0 if f_0 <= f_1 else a + ratio_1 * (b-a)
            x_init_0 = x.copy()
            x_init_1 = x.copy()
            x_init_0[which_var] = x_0
            x_init_1[which_var] = x_1
            f_0 = self.my_function(x_init_0) if f_0<=f_1 else f_1
            f_1 = f_0 if f_0<=f_1 else self.my_function(x_init_1)
            if b-a > tol_gr:
                pass
            else:
                return 0.5*(x_0+x_1)
        print('Golden Ratio Method Cannot Converge Within the Designated Iteration!')

    def Brent(self, x_0, which_var):
        iter_Brent = 10000
        tol_Brent = 0.001
        ax = x_0[which_var][0]
        bx = x_0[which_var][1]
        a = min(ax, bx)
        b = max(ax, bx)
        for i in range(iter_Brent):
            x_m = (a + b) / 2
            x_init_0 = x_0.copy()
            x_init_0[which_var]=x_m
            f_x = self.my_function(x_init_0)
            u = self.golden_ratio([a, b], x_0, which_var)
            x_init_u = x_0.copy()
            x_init_u[which_var]=u
            f_u = self.my_function(x_init_u)
            check_1 = u >= x_m
            check_2 = f_u <= f_x
            if check_2:
                if check_1:
                    a = x_m
                else:
                    b = x_m
            else:
                if not check_1:
                    a = u
                else:
                    b = u
            if b-a > tol_Brent:
                pass
            else:
                return 0.5*(a+b)
        print('Brent Method Cannot Converge Within the Designated Iteration!')
        print(which_var)
        print(b-a)

    def min_line_search(self):
        out_ls = []
        for i in range(0, self.n):
            in_ls = self.x_0.copy()
            in_ls[i] = self.x_range[i]
            u = self.Brent(in_ls, i)
            out_ls.append(u)
        return out_ls, self.my_function(out_ls)

if __name__ == "__main__":
    def my_function(x_0):
        x, y = x_0
        return 2 * (x ** 3) + 4 * (y ** 2) - 3 * y - 4 * x

    def dx_my_function(x_0):
        x, y = x_0
        return 6 * (x ** 2) - 4

    def dy_my_function(x_0):
        x, y = x_0
        return 8 * y - 3
    x_0 = [0, 0]
    x_range = np.array([[-1, 1], [-1, 1]])
    print(optim_non_drvt(x_0=x_0, x_range=x_range, my_function=my_function).min_line_search())

    # Compare result with scipy
    res = minimize(my_function, x_0, bounds=x_range, jac=[dx_my_function, dy_my_function])
    print(res.x)
    print(my_function(res.x))
