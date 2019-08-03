from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import pickle

def division_on_sets(x, t, tt, percent):
    num = x.shape[0]
    indexes = np.arange(num)
    ind_perm = np.random.permutation(indexes)
    ind_train = ind_perm[:np.int32(percent * num)]
    ind_test = ind_perm[np.int32(percent * num):]
    x_train = x[ind_train, :]
    x_test = x[ind_test, :]
    t_train = t[ind_train, :]
    t_test = t[ind_test, :]
    tt1 = tt[ind_train]
    tt2 = tt[ind_test]
    return x_train, x_test, t_train, t_test, tt1, tt2


def softmax(x):
    for i in range(x.shape[0]):
        x[i, :] -= np.max(x[i, :])
        temp = np.exp(x[i, :])
        x[i, :] = temp / sum(temp)
    return x      #np.exp(x) / np.sum(np.exp(x), axis=0)


def gradient_w(t, y, x, w, curr_lambda):
    return (y - t).transpose()@x + curr_lambda*w


def gradient_b(t, y):
    u = np.ones(t.shape[0])
    return (y - t).transpose()@u


def GD_stop(x, t, w_matrix, b_vector, gamma, curr_lambda, stop, x_test, t_for_E, t_for_T, k):
    y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
    grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
    grad_b = gradient_b(t, y)
    graph_1_1 = []
    graph_1_2 = []
    graph_1_3 = []
    graph_1_4 = []
    graph_1_5 = []
    i = 0
    acc = 0
    while np.linalg.norm(grad_w) >= stop:
        w_matrix = w_matrix - gamma*grad_w
        b_vector = b_vector - gamma*grad_b
        y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
        grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
        grad_b = gradient_b(t, y)

        y_test = softmax((w_matrix @ x_test.transpose()).transpose() + b_vector)
        cm_train = confusion_matrix(y, t_for_T, k)
        cm_test = confusion_matrix(y_test, t_for_E, k)
        graph_1_1.append(i)
        graph_1_2.append(accuracy(cm_train))
        graph_1_3.append(E_regulized(x, w_matrix, b_vector, t_for_T, curr_lambda, k))
        graph_1_4.append(accuracy(cm_test))
        graph_1_5.append(E_regulized(x_test, w_matrix, b_vector, t_for_E, curr_lambda, k))
        if i % 20 == 0:
            print("Итерация " + i.__str__() + ": точность(обуч.выборка) " + accuracy(cm_train).__str__() +
                  "     точность(обуч.выборка) " + accuracy(cm_test).__str__())
        i += 1
        if acc < accuracy(cm_test):
            acc = accuracy(cm_test)
            best_w = w_matrix
            best_b = b_vector
    return best_w, best_b, graph_1_1, graph_1_2, graph_1_3, graph_1_4, graph_1_5


def GD_steps(x, t, w_matrix, b_vector, gamma, curr_lambda, steps):
    y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
    grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
    grad_b = gradient_b(t, y)
    i = 0
    while i < steps:
        w_matrix = w_matrix - gamma*grad_w
        b_vector = b_vector - gamma*grad_b
        y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
        grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
        grad_b = gradient_b(t, y)
        i += 1
    return w_matrix, b_vector


def GD_difference_1(x, t, w_matrix, b_vector, gamma, curr_lambda, stop):
    y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
    grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
    grad_b = gradient_b(t, y)
    w_matrix_next = w_matrix - gamma*grad_w
    while np.linalg.norm(w_matrix_next - w_matrix) >= stop:
        w_matrix = w_matrix_next
        w_matrix_next = w_matrix - gamma*grad_w
        b_vector = b_vector - gamma*grad_b
        y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
        grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
        grad_b = gradient_b(t, y)
    return w_matrix, b_vector


def GD_difference_2(x, t, w_matrix, b_vector, gamma, curr_lambda, stop):
    epsilon0 = 0.00001
    y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
    grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
    grad_b = gradient_b(t, y)
    w_matrix_next = w_matrix - gamma*grad_w
    while np.linalg.norm(w_matrix_next - w_matrix) >= stop*(np.linalg.norm(w_matrix_next) + epsilon0):
        w_matrix = w_matrix_next
        w_matrix_next = w_matrix - gamma*grad_w
        b_vector = b_vector - gamma*grad_b
        y = softmax((w_matrix @ x.transpose()).transpose() + b_vector)
        grad_w = gradient_w(t, y, x, w_matrix, curr_lambda)
        grad_b = gradient_b(t, y)
    return w_matrix, b_vector


def onehotencoding(t, k):
    t_matrix = np.zeros((len(t), k), dtype=np.int32)
    for i in range(len(t)):
        t_matrix[i][t[i]] = 1
    return t_matrix


def E_regulized(x, w, b, t, curr_lambda, k):
    E = 0
    for i in range(x.shape[0]):
        e = 0
        for j in range(k):
            e += np.exp(w[j, :]@x[j, :].transpose() + b[j])
        e = np.log(e)
        E += e - (w[t[i], :]@x[t[i], :].transpose() + b[t[i]])
    E += curr_lambda*np.sum(w**2)
    return E


def standartization(x_matrix):
    for i in range(x_matrix.shape[1]):
        if np.std(x_matrix[:, i]) == 0:
            x_matrix[:, i] = 0
        else:
            x_matrix[:, i] = (x_matrix[:, i] - np.mean(x_matrix[:, i])) / np.std(x_matrix[:, i])
    return x_matrix


def confusion_matrix(y, t, k):
    cm = np.zeros((k, k))
    for i in range(y.shape[0]):
        curr_class = np.argmax(y[i, :])
        cm[t[i], curr_class] += 1
    return cm


def accuracy(cm):
    diag_sum = 0
    for i in range(cm.shape[0]):
        diag_sum += cm[i, i]
    acc = diag_sum / cm.sum()
    return acc


digits = load_digits()
x_matrix = digits.data
t = digits.target
t_matrix = onehotencoding(t, 10)

w_matrix = np.random.uniform(-0.0001, 0.0001, (10, x_matrix.shape[1]))
b_vector = np.random.uniform(-0.0001, 0.0001, 10)
gamma = 0.01
curr_lambda = 0.01
stop = 0.1

x_matrix = standartization(x_matrix)
x_train, x_test, t_train, t_test, t_for_T, t_for_E = division_on_sets(x_matrix, t_matrix, t, 0.8)

w_matrix, b_vector, gr1, gr2, gr3, gr4, gr5 = GD_stop(x_train, t_train, w_matrix, b_vector, gamma, curr_lambda, stop, x_test, t_for_E, t_for_T, 10)
'''w_matrix, b_vector = GD_stop(x_train, t_train, w_matrix, b_vector, gamma, curr_lambda, stop)
with open('C:/Users/Ралина/PycharmProjects/log_regression/w.pickle', 'wb') as f:
    pickle.dump(w_matrix, f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/b.pickle', 'wb') as f:
    pickle.dump(b_vector, f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/x_test.pickle', 'wb') as f:
    pickle.dump(x_test, f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/t_test_ohe.pickle', 'wb') as f:
    pickle.dump(t_test, f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/t_test.pickle', 'wb') as f:
    pickle.dump(t_for_E, f)'''      #решение методом сравнения нормы Е(w) с эпсилон 0.0001(для обучения)
'''with open('C:/Users/Ралина/PycharmProjects/log_regression/w.pickle', 'rb') as f:
    w_matrix = pickle.load(f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/b.pickle', 'rb') as f:
    b_vector = pickle.load(f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/x_test.pickle', 'rb') as f:
    x_test = pickle.load(f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/t_test_ohe.pickle', 'rb') as f:
    t_test = pickle.load(f)
with open('C:/Users/Ралина/PycharmProjects/log_regression/t_test.pickle', 'rb') as f:
    t_for_E = pickle.load(f)'''      #решение методом сравнения нормы Е(w) с эпсилон 0.0001(для тестирования)

E = E_regulized(x_test, w_matrix, b_vector, t_for_E, curr_lambda, 10)
y = softmax((w_matrix @ x_test.transpose()).transpose() + b_vector)
cm = confusion_matrix(y, t_for_E, 10)
print(E)
print(cm)
acc = accuracy(cm)
print("Точность на тестовой выборке = " + (acc * 100).__str__())

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(gr1, gr2, '-g')
plt.title("Зависимость точности от итерации при обучении")
plt.subplot(2, 2, 2)
plt.plot(gr1, gr3, '-b')
plt.title("Зависимость целевой функции от итерации при обучении")
plt.subplot(2, 2, 3)
plt.plot(gr1, gr4, '-g')
plt.title("Зависимость точности от итерации при тестировании")
plt.subplot(2, 2, 4)
plt.plot(gr1, gr5, '-b')
plt.title("Зависимость целевой функции от итерации при тестировании")
plt.show()
