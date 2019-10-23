import numpy as np
from itertools import combinations_with_replacement

feature = np.genfromtxt('./dataset_X.csv', delimiter=',')[:, 1:]
label = np.genfromtxt('./dataset_T.csv', delimiter=',')[:, -1]


def phi(X, M):
    # X 為原始input data, M為次方
    # 各次方的項數
    x_num = [1]
    phi_list = []
    # 產生排列組合(允許重複)
    for i in range(1, M+1):
        com = list(combinations_with_replacement(range(X.shape[1]), i))
        x_num.append(len(com)+x_num[i-1])
        phi_list.extend(com)

    # 建立Phi matrix, bias內建(M=0)
    phi_matrix = np.ones((len(X), len(phi_list)+1))
    x_num = np.asarray(x_num)

    # m筆data/ row
    for m in range(X.shape[0]):
        # 共幾個參數
        for n in range(len(phi_list)):
            sum = 1
            K = np.sort(np.argwhere(n+1 >= x_num))[-1]+1
            for k in range(K[0]):
                sum = sum * X[m][phi_list[n][k - 1]]
            # 第一col是bias不動
            phi_matrix[m, n+1] = sum
    return phi_matrix


# 計算不同order的PHI matrix
PHI = phi(feature, 1)
PHI2 = phi(feature, 2)


# Close form solution
def get_close_form(X, Y):
    """
    透過close form算出weight
    :param X: data的feature經過PHI轉換
    :param Y: data的target/ label
    :return:
    """
    # first = np.dot(X.T, X)
    # second = np.dot(X.T, Y)
    
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    # return np.dot(np.linalg.inv(first), second)


# M=1 時的close form weight, 取前877筆當train
w1 = get_close_form(phi(feature[:877], 1), label[:877])
w2 = get_close_form(phi(feature[:877], 2), label[:877])


# 1-a. 算 loss
def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


# train的loss
loss_w1 = rmse(phi(feature[:877], 1).dot(w1), label[:877])  # 3.93130...
loss_w2 = rmse(np.dot(phi(feature[:877], 2), w2), label[:877])  # 3.10346...

# todo: validation的 loss


rmse_combinations = []
# todo: 1-b. 挑feature
for i in range(17):
    tmp = np.delete(feature, i, axis=1)
    phi_m = phi(tmp[:877], 1)
    weights = get_close_form(phi_m, label[:877])
    rmse_tmp = rmse(phi_m.dot(weights), label[:877])
    rmse_combinations.append(rmse_tmp)

# todo: 2-a. 用選出的feature做polynomial (M>=3)

# todo: 2-b. 測試不同order成效


# 3-a.
def close_form_l2(X, Y, lamda):
    """
    透過close form加上L2 norm算出weight
    :param X: data的feature經過PHI轉換
    :param Y: data的target/ label
    :param lamda: L2的係數
    :return:
    """
    tmp = np.linalg.inv(X.T.dot(X) + lamda * np.identity(X.shape[1])).dot(X.T)
    return tmp.dot(Y)


w1_l2 = close_form_l2(phi(feature[:877], 1), label[:877], 0.001)
w2_l2 = close_form_l2(phi(feature[:877], 2), label[:877], 0.001)
