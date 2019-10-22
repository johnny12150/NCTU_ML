import numpy as np
from itertools import combinations_with_replacement

feature = np.genfromtxt('./HW1/dataset_X.csv', delimiter=',')[:, 1:]
label = np.genfromtxt('./HW1/dataset_T.csv', delimiter=',')[:, -1]


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
    print(x_num)

    # m筆data/ row
    for m in range(X.shape[0]):
        # 共幾個參數
        for n in range(len(phi_list)):
            sum = 1
            for k in range(1, M+1):
                if n < x_num[k]:
                    print(k)
                    print(phi_list[n])
                    if k > len(phi_list[n]):
                        break
                    sum = sum*feature[m][phi_list[n][k-1]]
            # 第一col是bias不動
            phi_matrix[m, n+1] = sum
    return phi_matrix


PHI = phi(feature, 1)
PHI2 = phi(feature, 2)


# Close form solution
def get_close_form(X, Y):
    # first = np.dot(X.T, X)
    # second = np.dot(X.T, Y)
    
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    # return np.dot(np.linalg.inv(first), second)


# M=1 時的close form weight
w1 = get_close_form(phi(feature[:877], 1), label[:877])
w2 = get_close_form(phi(feature[:877], 2), label[:877])
