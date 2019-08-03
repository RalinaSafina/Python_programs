from sklearn.datasets import load_digits
import numpy as np


def division_on_sets(x, t, percent):
    num = x.shape[0]
    indexes = np.arange(num)
    ind_perm = np.random.permutation(indexes)
    ind_train = ind_perm[:np.int32(percent * num)]
    ind_valid = ind_perm[np.int32(percent * num):np.int32(((1-percent)/2 + percent) * num)]
    ind_test = ind_perm[np.int32(((1-percent)/2 + percent) * num):]
    x_train = x[ind_train, :]
    x_test = x[ind_test, :]
    t_train = t[ind_train, :]
    t_test = t[ind_test, :]
    x_valid = x[ind_valid, :]
    t_valid = t[ind_valid, :]
    return x_train, x_test, x_valid, t_train, t_test, t_valid


def bagging(x_train, t_train):
    num = x_train.shape[0]
    indexes = np.arange(num)
    ind_perm = np.random.permutation(indexes)
    start = np.random.randint(0, len(ind_perm)/2)
    stop = np.random.randint(start + np.random.randint(0, len(ind_perm)/2), len(ind_perm))
    x = x_train[ind_perm[start:stop], :]
    t = t_train[ind_perm[start:stop], :]
    return x, t


def onehotencoding(t, k):
    t_matrix = np.zeros((len(t), k), dtype=np.int32)
    for i in range(len(t)):
        t_matrix[i][t[i]] = 1
    return t_matrix


def toclassvalues(t):
    tt = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        tt[i] = np.argmax(t[i, :])
    return tt


class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_index = None
        self.split_value = None
        self.entropy_value = None
        self.terminal_class = []


class DT:   #decision tree
    def __init__(self,
                 max_depth,
                 min_entropy,
                 min_elements):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elements = min_elements
        self.root = Node()
        self.root.depth = 1


def CountRootEntropy(root, t_train):
    entropy = np.zeros(t_train.shape[1])
    for i in range(t_train.shape[1]):
        entropy[i] = sum(row[i] for row in t_train)
    summ = 0
    for i in range(len(entropy)):
        summ += entropy[i] / np.sum(entropy) * np.log(entropy[i] / np.sum(entropy))
    summ *= (-1)
    root.entropy_value = summ


def CountEntropy(t_array, indexes):
    entropy = np.zeros(t_array.shape[1])
    t = t_array[indexes, :]
    entropy_value = 0
    for i in range(t.shape[1]):
        entropy[i] = sum(row[i] for row in t)
    for i in range(len(entropy)):
        if np.sum(entropy) > 0 and entropy[i] != 0:
            entropy_value += entropy[i] / np.sum(entropy) * np.log(entropy[i] / np.sum(entropy))
    entropy_value *= (-1)
    return entropy_value


def RootGini(root, t):
    gini = np.zeros(t.shape[1])
    for i in range(t.shape[1]):
        gini[i] = sum(row[i] for row in t)
    summ = 0
    for i in range(len(gini)):
        summ += (gini[i] / np.sum(gini))**2
    summ = 1 - summ
    root.entropy_value = summ


def Gini(t_array, indexes):
    gini = np.zeros(10)
    t = t_array[indexes, :]
    gini_value = 0
    for i in range(t.shape[1]):
        gini[i] = sum(row[i] for row in t)
    for i in range(len(gini)):
        if np.sum(gini) > 0 and gini[i] != 0:
            gini_value += (gini[i] / np.sum(gini))**2
    gini_value = 1 - gini_value
    return gini_value


def BuildTree(root, x_train, t_train, depth, max_depth, min_entropy, min_elements, unceirtain):
    best_inf_gain = 0
    best_left = []
    best_right = []
    best_tau2 = None
    best_psi = None
    for psi in range(x_train.shape[1]):
        for tau_2 in range(int(np.min(x_train)), int(np.max(x_train))):
            left_ind = []
            right_ind = []
            for i in range(0, x_train.shape[0]):
                if x_train[i][psi] < tau_2:
                    left_ind.append(i)
                else:
                    right_ind.append(i)
            if unceirtain == "entropy":
                inf_gain = root.entropy_value - len(left_ind) / x_train.shape[0] * CountEntropy(t_train, left_ind) - \
                                   len(right_ind) / x_train.shape[0] * CountEntropy(t_train, right_ind)
            elif unceirtain == "gini":
                inf_gain = root.entropy_value - len(left_ind) / x_train.shape[0] * Gini(t_train, left_ind) - \
                                   len(right_ind) / x_train.shape[0] * Gini(t_train, right_ind)
            if inf_gain > best_inf_gain:
                best_left = left_ind
                best_right = right_ind
                best_inf_gain = inf_gain
                best_psi = psi
                best_tau2 = tau_2

    root.right = Node()
    root.left = Node()
    if unceirtain == "entropy":
        root.right.entropy_value = CountEntropy(t_train, best_right)
        root.left.entropy_value = CountEntropy(t_train, best_left)
    elif unceirtain == "gini":
        root.right.entropy_value = Gini(t_train, best_right)
        root.left.entropy_value = Gini(t_train, best_left)
    root.split_index = best_psi
    root.split_value = best_tau2
    depth += 1
    if depth < max_depth and root.left.entropy_value > min_entropy and len(best_left) > min_elements:
        BuildTree(root.left, x_train[best_left, :], t_train[best_left, :], depth, max_depth, min_entropy, min_elements, unceirtain)
    else:
        root.left.terminal_class = class_vector(toclassvalues(t_train[best_left, :]))
    if depth < max_depth and root.right.entropy_value > min_entropy and len(best_right) > min_elements:
        BuildTree(root.right, x_train[best_right, :], t_train[best_right, :], depth, max_depth, min_entropy, min_elements, unceirtain)
    else:
        root.right.terminal_class = class_vector(toclassvalues(t_train[best_right, :]))


def split_in_node(root, x):
    if root.terminal_class == []:
        if x[root.split_index] < root.split_value:
            return split_in_node(root.left, x)
        else:
            return split_in_node(root.right, x)
    else:
        return root.terminal_class


def classification(tree, x_test, test_matrix):
    for i in range(x_test.shape[0]):
        test_matrix[i] += split_in_node(tree.root, x_test[i])


def accuracy(cm):
    diag_sum = 0
    for i in range(cm.shape[0]):
        diag_sum += cm[i, i]
    acc = diag_sum / cm.sum()
    return acc


def build_conf_matrix(x_test, Test_matrix, t_test, conf_matrix):
    for i in range(x_test.shape[0]):
        predicted_class = np.argmax(Test_matrix[i, :])
        real_class = np.argmax(t_test[i, :])
        conf_matrix[real_class, predicted_class] += 1


def class_vector(t):
    tt = np.zeros(10)
    for j in range(0, 10):
        amount = 0
        for i in range(0, len(t)):
            if t[i] == j:
                amount += 1
        if len(t) != 0:
            tt[j] = amount / len(t)
    return tt


digits = load_digits()
x_matrix = digits.data
t = digits.target
number_of_classes = 10
t_matrix = onehotencoding(t, 10)
x_train, x_test, x_valid, t_train, t_test, t_valid = division_on_sets(x_matrix, t_matrix, 0.7)

best_valid_accuracy = 0
number_of_trees = None
random_forest = []
for_valid_cm = None
iterations = np.random.randint(1, 10)
for j in range(0, iterations):
    print(j)
    num_of_trees = np.random.randint(1, 20)
    rf = []
    for i in range(0, num_of_trees):
        x, tt = bagging(x_train, t_train)
        tree = DT(max_depth=np.random.randint(10, 20), min_entropy=np.random.random() * 0.2,
                  min_elements=np.random.randint(number_of_classes, 2 * number_of_classes))
        unceirtain = np.random.choice(['entropy', 'gini'])
        if unceirtain == 'entropy':
            CountRootEntropy(tree.root, tt)
            BuildTree(tree.root, x, tt, 1, tree.max_depth, tree.min_entropy, tree.min_elements, "entropy")
        else:
            RootGini(tree.root, tt)
            BuildTree(tree.root, x, tt, 1, tree.max_depth, tree.min_entropy, tree.min_elements, "gini")
        rf.append(tree)
    valid_matrix = np.zeros((x_valid.shape[0], number_of_classes))
    for i in range(0, num_of_trees):
        classification(rf[i], x_valid, valid_matrix)
    valid_matrix /= num_of_trees
    vm = np.zeros((number_of_classes, number_of_classes), dtype=np.int32)
    build_conf_matrix(x_valid, valid_matrix, t_valid, vm)
    valid_accuracy = accuracy(vm)
    if valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        for_valid_cm = valid_matrix
        random_forest = rf
        number_of_trees = num_of_trees

vm = np.zeros((number_of_classes, number_of_classes), dtype=np.int32)
build_conf_matrix(x_valid, for_valid_cm, t_valid, vm)

print("VALIDATION SET")
print(vm)
print(best_valid_accuracy)

train_matrix = np.zeros((x_train.shape[0], number_of_classes))
for i in range(0, number_of_trees):
    classification(random_forest[i], x_train, train_matrix)
train_matrix /= number_of_trees
cm = np.zeros((number_of_classes, number_of_classes), dtype=np.int32)
build_conf_matrix(x_train, train_matrix, t_train, cm)
train_accuracy = accuracy(cm)

print("TRAIN SET")
print(cm)
print(train_accuracy)

Test_matrix = np.zeros((x_test.shape[0], number_of_classes))
for i in range(0, number_of_trees):
    classification(random_forest[i], x_test, Test_matrix)
Test_matrix /= number_of_trees
conf_matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.int32)
build_conf_matrix(x_test, Test_matrix, t_test, conf_matrix)
accuracy = accuracy(conf_matrix)

print("TEST SET")
print(conf_matrix)
print(accuracy)
