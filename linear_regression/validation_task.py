import numpy as np
import math
import matplotlib.pyplot as plt

lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]


def basic_funcs(n, x, DM):
    m_temp = np.empty((len(x), 1))
    if n >= 0 and n <= 10:
        for i in range(0, len(x)):
            m_temp[i, 0] = x[i] ** n
    if n == 11:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.sin(x[i])
    if n == 12:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.cos(x[i])
    if n == 13:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.exp(x[i])
    if n == 14:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.tan(x[i])
    if n == 15:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.sinh(x[i])
    if n == 16:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.cosh(x[i])
    if n == 17:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.sin(x[i] * 2)
    if n == 18:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.sin(x[i] / 2)
    if n == 19:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.cos(x[i] * 3)
    if n == 20:
        for i in range(0, len(x)):
            m_temp[i, 0] = math.exp(x[i] / 4)
    DM = np.hstack((DM, m_temp))
    return DM


def basic_funcs_print(basic_funcs_inds):
    temp = []
    for k in range(len(basic_funcs_inds)):
        n = basic_funcs_inds[k]
        if n >= 0 and n <= 10:
            temp.append("to power of " + str(n))
        if n == 11:
            temp.append("sin(x)")
        if n == 12:
            temp.append("cos(x)")
        if n == 13:
            temp.append("exp(x)")
        if n == 14:
            temp.append("tan(x)")
        if n == 15:
            temp.append("sinh(x)")
        if n == 16:
            temp.append("cosh(x)")
        if n == 17:
            temp.append("sin(2*x)")
        if n == 18:
            temp.append("sin(x/2)")
        if n == 19:
            temp.append("cos(3*x)")
        if n == 20:
            temp.append("exp(x/4)")
    print(temp)


def data_generation(data_number):
    x = np.linspace(0, 1, data_number)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    e = 10 * np.random.randn(data_number)
    t = z + e
    return x, z, t


def division_on_sets(x, t, percent):
    indexes = np.arange(len(t))
    ind_perm = np.random.permutation(indexes)
    ind_train = ind_perm[:np.int32(percent * len(t))]
    ind_valid = ind_perm[np.int32(percent * len(t)):np.int32((percent + (1 - percent) / 2) * len(t))]
    ind_test = ind_perm[np.int32((percent + (1 - percent) / 2) * len(t)):]
    x_train = x[ind_train]
    x_valid = x[ind_valid]
    x_test = x[ind_test]
    t_train = t[ind_train]
    t_valid = t[ind_valid]
    t_test = t[ind_test]
    return x_train, x_valid, x_test, t_train, t_valid, t_test


def get_design_matrix(basic_func_ind, x_set):
    DM = np.empty((len(x_set), 0))
    for k in range(0, len(basic_func_ind)):
        DM = basic_funcs(basic_func_ind[k], x_set, DM)
    return DM


def get_w(DM, llambda, t_set):
    I = np.eye((DM.transpose() @ DM).shape[1])
    w = np.linalg.inv(DM.transpose() @ DM + llambda * I) @ DM.transpose() @ t_set.transpose()
    return w


def get_error(w, DM, t_set, llambda):
    y = w.transpose() @ DM.transpose()
    E = 0
    for i in range(0, len(t_set)):
        E += (y[i] - t[i]) ** 2
    E /= 2
    temp = 0
    for i in range(0, len(w)):
        temp += w[i] ** 2
    E_new = E + llambda / 2 * temp
    return E_new


x, z, t = data_generation(1000)
x_train, x_valid, x_test, t_train, t_valid, t_test = division_on_sets(x, t, 0.8)
N = 1000
E_min = 10 ** 10
for i in range(N):
    curr_lambda = np.random.choice(lambdas)
    curr_basic_func_inds = np.random.choice(np.arange(21), 6, replace=False)
    DM_train = get_design_matrix(curr_basic_func_inds, x_train)
    w_curr = get_w(DM_train, curr_lambda, t_train)
    DM_valid = get_design_matrix(curr_basic_func_inds, x_valid)
    E_curr = get_error(w_curr, DM_valid, t_valid, curr_lambda)
    if E_curr < E_min:
        E_min = E_curr
        best_lambda = curr_lambda
        best_func_inds = curr_basic_func_inds
        w_best = w_curr
DM_test = get_design_matrix(best_func_inds, x_test)
E_test = get_error(w_best, DM_test, t_test, best_lambda)
y = (w_best.transpose() @ DM_test.transpose())
print("Error = ", E_test)
print("Best lambda = ", best_lambda)
basic_funcs_print(best_func_inds)

result = sorted(zip(x_test, y), reverse=False)
x_test = [e[0] for e in result]
y = [e[1] for e in result]

plt.figure()
plt.plot(x, t, 'r,')
plt.plot(x, z, '-g')
plt.plot(x_test, y, '-b')
plt.show()
