import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import random

feature = np.genfromtxt('./dataset_X.csv', delimiter=',')[:, 1:]
label = np.genfromtxt('./dataset_T.csv', delimiter=',')[:, -1]


def phi(X, M):
    # X 為原始input data, M為次方
    # 各次方的項數
    x_num = [1]
    phi_list = []
    if len(X.shape) > 1:
        # 產生排列組合(允許重複)
        for i in range(1, M+1):
            com = list(combinations_with_replacement(range(X.shape[1]), i))
            x_num.append(len(com)+x_num[i-1])
            phi_list.extend(com)
    else:
        # 加入基本的bias
        phi_list.append(1)

    # 建立Phi matrix, bias內建(M=0)
    phi_matrix = np.ones((len(X), len(phi_list)+1))
    x_num = np.asarray(x_num)

    if len(X.shape) > 1:
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
    else:
        # 處理單一個feature
        phi_matrix[:, 1] = X
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
    
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)


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

    plt.plot(x_range, train, label='train')
    plt.plot(x_range, val, label='validation')
    plt.legend(loc='best')
    plt.xlabel(xtitle)
    plt.xticks(x_range)
    plt.ylabel('RMSE')
    plt.show()
    plt.close()


def kmeans(sample, K, maxiter):
    """
    :param sample: 要分群的data
    :param K: 分幾群
    :param maxiter: 最多跑幾次
    :return: 回傳Center, 屬於哪群, 跟with-in-class-distance
    """
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K, D))  # K個中心點/K群的座標
    L = np.zeros((N, 1))  # label
    L1 = np.zeros((N, 1))
    dist = np.zeros((N, K))
    idx = random.sample(range(N), K)
    C = sample[idx, :]
    iter = 0
    while (iter < maxiter):
        # K個中心點分開算
        for i in range(K):
            # np.tile會repeat, 每個中心點repeat N*1次
            dist[:, i] = np.sum((sample - np.tile(C[i, :], (N, 1))) ** 2, axis=1)
        L1 = np.argmin(dist, 1)
        if (iter > 0 and np.array_equal(L, L1)):
            print(iter)
            break
        L = L1
        for i in range(K):
            # 取非0的index
            idx = np.nonzero(L == i)[0]
            if (len(idx) > 0):
                C[i, :] = np.mean(sample[idx, :], 0)
        iter += 1

    # C[L,:] -> K類各自center
    wicd = np.sum(np.sqrt(np.sum((sample - C[L, :]) ** 2, 1)))
    return C, L, wicd


# 2-a. Gaussian basis
def gaussian_basis_function(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def sigmoid_basis(x, mu, sigma=0.1):
    return 1/ (1 - np.exp((x - mu)/ sigma))


# 產生gaussian的phi
def phi_basis(X, sigma=0, method='gaussian'):
    phi = np.zeros(X.shape)
    # normalize
    # X = (X-np.mean(X, 0) / np.std(X, 0))
    for i in range(X.shape[1]):
        if method == 'gaussian':
            if sigma:
                phi[:, i] = gaussian_basis_function(X[:, i], X[:, i].mean(), sigma)
            else:
                phi[:, i] = gaussian_basis_function(X[:, i], X[:, i].mean())
        elif method == 'sigmoid':
            if sigma:
                phi[:, i] = sigmoid_basis(X[:, i], X[:, i].mean(), sigma)
            else:
                # 直接帶該feature的sigma看看
                phi[:, i] = sigmoid_basis(X[:, i], X[:, i].std())
    return phi


# 3-a.
def close_form_l2(X, Y, lamda):
    # https://xavierbourretsicotte.github.io/intro_ridge.html
    """
    透過close form加上L2 norm算出weight
    :param X: data的feature經過PHI轉換
    :param Y: data的target/ label
    :param lamda: L2的係數
    :return:
    """
    tmp = np.linalg.inv(X.T.dot(X) + lamda * np.eye(X.shape[1])).dot(X.T)
    return tmp.dot(Y)


def train_vali(X_train, y_train, X_test, y_test, order, l2=0, cross=1, basis='linear', sigma=0):
    """
    算出train跟validation的loss
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param order: 幾次方
    :param l2: 設定lambda
    :param cross: cross_validation要切幾分
    :param basis: 使用的基底函數
    :return: weight, train跟validation的loss
    """

    train_l = 0
    val_l = 0
    X = X_train.copy()
    Y = y_train.copy()
    for i in range(cross):
        if cross !=1:
            # 手動切train test
            parts = len(X) / cross
            idx = range(int(parts * i), int(parts * (i+1)))
            y_test = Y[idx]
            y_train = Y[list(set(range(len(Y))) - set(idx))]
            X_test = X[idx]
            X_train = X[list(set(range(len(Y))) - set(idx))]

        if basis!='linear':
            if basis =='sigmoid':
                phi_train = phi_basis(X_train, sigma, 'sigmoid')
                phi_test = phi_basis(X_test, sigma, 'sigmoid')
            else:
                phi_train = phi_basis(X_train, sigma)
                phi_test = phi_basis(X_test, sigma)
        else:
            phi_train = phi(X_train, order)
            phi_test = phi(X_test, order)
        if l2:
            w = close_form_l2(phi_train, y_train, l2)
        else:
            # 計算weight
            w = get_close_form(phi_train, y_train)

        # 計算train, validation loss
        train_l += rmse(phi_train.dot(w), y_train)
        val_l += rmse(phi_test.dot(w), y_test)

    train_l = train_l / cross
    val_l = val_l / cross
    return w, train_l, val_l


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
rmse_val = []
# 1-b. 挑feature
for i in range(17):
    tmp = np.delete(feature, i, axis=1)
    w, t, v = train_vali(tmp[:877], label[:877], tmp[877:], label[877:], 1)
    rmse_trian.append(t)
    rmse_val.append(v)

plot_loss(rmse_trian, rmse_val, range(len(rmse_trian)), 'Column of feature droped')

# 只使用一個feature來選特徵
rmse_trian = []
rmse_val = []
# 1-b. 挑feature
for i in range(17):
    tmp = feature[:, i]
    w, t, v = train_vali(tmp[:877], label[:877], tmp[877:], label[877:], 1)
    rmse_trian.append(t)
    rmse_val.append(v)

plot_loss(rmse_trian, rmse_val, range(len(rmse_trian)), 'Column of feature selected')

rmse_trian = []
rmse_val = []
# 2-a. 測試不同order成效
for m in range(10):
    # 2-b. 用選出的feature做polynomial, 切4分做cross-validation
    w, t, v = train_vali(feature[:, [2, 8, 13]], label, feature[:, [2, 8, 13]], label, m, 0, 8)
    rmse_trian.append(t)
    rmse_val.append(v)

plot_loss(rmse_trian, rmse_val, range(1, len(rmse_trian)+1), 'Different order for polynomial')

# 透過kmeans找center
C, L, wicd = kmeans(feature, 17, 10000)
# gaussian
rmse_trian = []
rmse_val = []
sigma_list = [0.001, 0.1, 1, 10, 1000]
for i, si in enumerate(sigma_list):
    if si > 0.01:
        w, t, v = train_vali(feature[:, [2, 8, 13]], label, feature[:, [2, 8, 13]], label, 0, 0, 4, 'gaussian', si)
        rmse_trian.append(t)
        rmse_val.append(v)
    else:
        rmse_trian.append('Singular Matrix')
        rmse_val.append('Singular Matrix')

# sigmoid
rmse_trian = []
rmse_val = []
for i, si in enumerate(sigma_list):
    w, t, v = train_vali(feature[:, [2, 8, 13]], label, feature[:, [2, 8, 13]], label, 0, 0, 4, 'sigmoid', si)
    rmse_trian.append(t)
    rmse_val.append(v)


# 3-b. 測試不同lamda
lamda_list = [0.001, 0.01, 0.1, 1, 10, 100]
for l, lam in enumerate(lamda_list):
    rmse_trian = []
    rmse_val = []
    for m in range(5):
        # polynomial
        w, t, v = train_vali(feature[:, [2, 8, 13]], label, feature[:, [2, 8, 13]], label, m, lam, 4)
        # 選其他feature
        # w, t, v = train_vali(feature[:, [1, 2, 3, 8]], label, feature[:, [1, 2, 3, 8]], label, m, lam, 8)

        rmse_trian.append(t)
        rmse_val.append(v)

    # 比較 w1, w2各自的 train, test loss
    # plot_loss(rmse_trian, rmse_val, range(1, len(rmse_trian)+1), 'Lambda= ' + str(lam) + ' L2 polynomial')

rmse_trian = []
rmse_val = []
for l, lam in enumerate(lamda_list):
    # gaussian
    w, t, v = train_vali(feature[:, [2, 8, 13]], label, feature[:, [2, 8, 13]], label, 0, lam, 4, 'gaussian', 1000)
    # sigmoid
    # w, t, v = train_vali(feature[:, [2, 8, 13]], label, feature[:, [2, 8, 13]], label, 0, lam, 4, 'sigmoid', 0.01)
    rmse_trian.append(t)
    rmse_val.append(v)

