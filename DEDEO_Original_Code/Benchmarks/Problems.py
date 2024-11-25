import numpy as np
from categorical_set import my_categorical_set

class Problems(object):
    def __init__(self, prob_name, dim_c, dim_d):
        '''
        :param prob_name: problem name, including "ellipsoid", "rosenbrock",
        "ackley","griewank","rastrigin","sphere","weierstrass","Lunacek","CF1","CF4"

        :param dim_r: dimension of continuous decision variables, including 5, 15, 25 for "ellipsoid", "rosenbrock",
        "ackley","griewank","rastrigin"; 5, 15 for "sphere","weierstrass","Lunacek","CF1","CF4".
        :param dim_d: dimension of discrete decision variables, including 5, 15, 25 for "ellipsoid", "rosenbrock",
        "ackley","griewank","rastrigin"; 5, 15 for "sphere","weierstrass","Lunacek","CF1","CF4".
        :param N_d: the number of values for discrete variables
        '''
        self.r = dim_c
        self.o = dim_d
        self.dim = dim_c + dim_d
        if len(prob_name):
            f, v_dv, N_lst, _ = my_categorical_set(prob_name,  self.dim)
        else:
            raise NotImplementedError
        self.N_d = N_lst[0] #每个位置的类变量集合的数目都相等
        self.v_dv = v_dv
        self.N_lst = N_lst
        self.bounds = [-100, 100]
        self.F = f
        # self.x_shift = np.loadtxt("Benchmarks/shift_data/data_" + prob_name + '.txt')[:self.dim].reshape(1, -1)
        self.M = np.loadtxt("Benchmarks/shift_data/data_elliptic_M_D10" '.txt')[:self.dim].reshape(10, 10)
        # def F1(X):
        #     # X 是连续变量与类变量的组合
        #     x0 = [7.7624, -51.0984, -95.5110, -68.7425, 8.7344, 0.0577, -36.7734, 44.3837, 99.8131, -12.1793]
        #
        #     z1 = X-x0
        #     z2 = np.dot(z1, self.M)
        #     z = z2**2
        #     y = z.sum()
        #
        #     return y.astype(float)

        # if prob_name == "F1":
        #     self.F = F1