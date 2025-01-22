import numpy as np
from scipy.optimize import minimize

class optim_non_drvt:
    # This class collects optimization methods when f(x) doesn't have a derivative

    def __init__(self, x_0, x_range, my_function, mat):
        self.x_0 = x_0
        self.x_range = x_range
        self.n = len(x_0)
        self.my_function = my_function
        np.random.seed(10)
        # self.mat = np.ones((self.n, self.n))
        # self.mat = np.maximum(np.random.normal(0, 1, (self.n, self.n)), 0)
        self.mat = mat

    def golden_ratio(self, a_b, x, which_var, tol_gr):
        ratio_1 = 0.618
        ratio_2 = 1 - ratio_1
        iter_gr = 10000
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

    def Brent(self, x_0, which_var, tol_Brent, tol_gr):
        iter_Brent = 10000
        ax = x_0[which_var][0]
        bx = x_0[which_var][1]
        a = min(ax, bx)
        b = max(ax, bx)
        for i in range(iter_Brent):
            x_m = (a + b) / 2
            x_init_0 = x_0.copy()
            x_init_0[which_var]=x_m
            f_x = self.my_function(x_init_0)
            u = self.golden_ratio([a, b], x_0, which_var, tol_gr)
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

    def min_line_search(self, x_0, tol_Brent, tol_gr):
        out_ls = []
        for i in range(0, self.n):
            in_ls = x_0.copy()
            in_ls[i] = self.x_range[i]
            u = self.Brent(in_ls, i, tol_Brent, tol_gr)
            out_ls.append(u)
        return out_ls, self.my_function(out_ls)

    def mod_Powell(self):
        iter_Powell = 1000
        tol_Powell = 1
        tol_Brent = 0.11
        tol_gr = 0.11
        x_init = self.x_0.copy()
        f_init = self.my_function(x_init)
        mat = self.mat.copy()
        for j in range(iter_Powell):
            print('Loop')
            print(j)
            x_last = x_init.copy()
            f_x_last = f_init
            delta = 0
            for k in range(0, len(mat)):
                mat_x = mat[k].copy()
                minimized = self.min_line_search(x_0=[x + y for x, y in zip(x_last, mat_x)], tol_Brent=tol_Brent, tol_gr=tol_gr)
                x, f_x = minimized
                if f_x_last - f_x > delta:
                    delta = f_x_last - f_x
            f_x_last = f_x
            x_last = x
            print("Best x is ...")
            print(x_last)
            if f_x_last - f_init > tol_Powell:
                print("Current Diff is")
                print(f_x_last - f_init)
                pass
            else:
                print("Initial f(x) is ...")
                print(f_init)
                print("Best f(x) is ...")
                print(f_x_last)
                return x, self.my_function(x)
            f_extra = self.my_function([2 * x - y for x, y in zip(x_last, x_init)])
            check_1 = f_extra >= f_init
            check_2 = (f_init - 2 * f_x_last + f_extra) * (f_init - f_x_last - delta) ** 2 >= delta * (
                        f_init - f_extra) ** 2
            if check_1 or check_2:
                x_init = x_last
                tol_gr = tol_gr - 0.0001
                tol_Brent = tol_Brent - 0.0001
            else:
                epsilon = [x - y for x, y in zip(x_last, x_init)]
                minimized = self.min_line_search(x_0=[x + y for x, y in zip(x_last, epsilon)], tol_Brent=tol_Brent, tol_gr=tol_gr)
                x, f_x = minimized
                x_lambda = [(x - y) / z for x, y, z in zip(x, x_last, epsilon)]
                mat_temp = mat.tolist()
                mat_temp.append(x_lambda)
                mat = mat_temp
                x_init = x
        print('Modified Powell Method Cannot Converge Within the Designated Iteration!')


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
    print(optim_non_drvt(x_0=x_0, x_range=x_range, my_function=my_function, mat='NA').min_line_search(x_0=x_0, tol_Brent=0.001, tol_gr=0.001))

    # Compare result with scipy
    res = minimize(my_function, x_0, bounds=x_range, jac=[dx_my_function, dy_my_function])
    print(res.x)
    print(my_function(res.x))
