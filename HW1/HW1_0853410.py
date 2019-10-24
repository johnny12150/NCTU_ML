import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

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
    """
    :param a: train/ validation
    :param b: validation/ train
    :return: rmse計算結果
    """
    return np.sqrt(np.mean(np.square(a - b)))


def plot_loss(train, val, x_range, xtitle='order change'):
    """
    畫train跟validation loss的圖
    :param train: train的RMSE結果
    :param val: validation的RMSE結果
    :param x_range: x軸的range
    :param xtitle: 圖片的x軸名稱
    """
    # x_range = range(1, len(train)+1)
    plt.plot(x_range, train, label='train')
    plt.plot(x_range, val, label='validation')
    plt.legend(loc='best')
    plt.xlabel(xtitle)
    plt.xticks(x_range)
    plt.ylabel('RMSE')
    plt.show()
    plt.close()


# train的loss
train_loss_w1 = rmse(phi(feature[:877], 1).dot(w1), label[:877])  # 3.93130...
train_loss_w2 = rmse(np.dot(phi(feature[:877], 2), w2), label[:877])  # 3.10346...

val_phi1 = phi(feature[877:], 1)
val_phi2 = phi(feature[877:], 2)
# validation的 loss
val_loss_1 = rmse(val_phi1.dot(w1), label[877:])  # 5.55501
val_loss_2 = rmse(val_phi2.dot(w2), label[877:])  # 6.86332

plot_loss([train_loss_w1, train_loss_w2], [val_loss_1, val_loss_2], range(1, 3))

rmse_trian = []
rmse_combinations = []
# todo: 1-b. 挑feature
for i in range(17):
    tmp = np.delete(feature, i, axis=1)
    phi_m = phi(tmp[:877], 1)
    weights = get_close_form(phi_m, label[:877])
    phi_test = phi(tmp[877:], 1)
    rmse_trian.append(rmse(phi_m.dot(weights), label[:877]))
    rmse_combinations.append(rmse(phi_test.dot(weights), label[877:]))

plot_loss(rmse_trian, rmse_combinations, range(len(rmse_trian)), 'Column of feature droped')

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


# 3-b. 測試不同lamda
lamda_list = [0.001, 0.01, 0.1, 1]
rmse_w1_l2 = []
rmse_w2_l2 = []
for l, lam in enumerate(lamda_list):
    w1_l2 = close_form_l2(phi(feature[:877], 1), label[:877], lam)
    w2_l2 = close_form_l2(phi(feature[:877], 2), label[:877], lam)
    test_phi1 = phi(feature[877:], 1)
    test_phi2 = phi(feature[877:], 2)
    rmse_w1_l2.append(rmse(test_phi1.dot(w1_l2), label[877:]))
    rmse_w2_l2.append(rmse(test_phi2.dot(w2_l2), label[877:]))

