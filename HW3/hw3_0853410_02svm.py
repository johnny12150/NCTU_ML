import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from collections import Counter

classes = [(0, 1), (0, 2), (1, 2)]


def poly_phi(x):
    if len(x.shape) == 1:
        return np.vstack((x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2)).T
    else:
        return np.vstack((x[:, 0] ** 2, np.sqrt(2) * x[:, 0] * x[:, 1], x[:, 1] ** 2)).T


def kernel_function(xn, xm):
    return np.dot(xn, xm)


def custom_svm(X, y, type_='linear', vs='ovo', c=1):
    if type_ == 'linear':
        clf = SVC(kernel='linear', C=c, decision_function_shape=vs)
    else:
        clf = SVC(kernel='poly', C=c, degree=2, decision_function_shape=vs)
    clf.fit(X, y)
    return clf  # return support vectors for plot


def update_weight(a, t, x, c=1, type_='linear'):
    at = a * t  # PRML 7.29
    if type_ == 'linear':
        w = at.dot(x)
    else:
        w = at.dot(poly_phi(x))
    # PRML 7.37
    M_indexes = np.where(((a > 0) & (a < c)))[0]
    S_indexes = np.nonzero(a)[0]
    Nm = len(M_indexes)

    if Nm == 0:
        b = -1
    else:
        #   b = np.mean(t[M_indexes] - np.linalg.multi_dot([at[S_indexes], x[S_indexes], x[M_indexes].T]))
        if type_ == 'linear':
            b = np.mean(t[M_indexes] - at[S_indexes].dot(kernel_function(x[M_indexes], x[S_indexes].T).T))
        else:
            b = np.mean(t[M_indexes] - at[S_indexes].dot(kernel_function(poly_phi(x[M_indexes]), poly_phi(x[S_indexes]).T).T))
    return w, b


def train_svm(X, labels, support_vectors, coef, type_='linear'):
    size = 100  # data points for each  class
    # 1. prepare params
    target_dict = {}  # target
    target_dict[(0, 1)] = np.concatenate((np.ones(size), np.full([size], -1), np.zeros(size)))
    target_dict[(0, 2)] = np.concatenate((np.ones(size), np.zeros(size), np.full([size], -1)))
    target_dict[(1, 2)] = np.concatenate((np.zeros(size), np.ones(size), np.full([size], -1)))

    # multiplier
    multiplier = np.zeros([len(X), 2])
    multiplier[support_vectors] = coef.T
    multiplier_dict = {}
    multiplier_dict[(0, 1)] = np.concatenate((multiplier[:size * 2, 0], np.zeros(size)))
    multiplier_dict[(0, 2)] = np.concatenate((multiplier[:size, 1], np.zeros(size), multiplier[size * 2:, 0]))
    multiplier_dict[(1, 2)] = np.concatenate((np.zeros(size), multiplier[size:, 1]))

    # 2. train weight and bias for each class
    weights = {}
    biases = {}

    for c1, c2 in labels:
        if not type_=='linear':
            weight, bias = update_weight(multiplier_dict[(c1, c2)], target_dict[(c1, c2)], X, type_='poly')
        else:
            weight, bias = update_weight(multiplier_dict[(c1, c2)], target_dict[(c1, c2)], X)
        weights[(c1, c2)] = weight
        biases[(c1, c2)] = bias

    return weights, biases


def make_prediction(X, label, weight_dict, bias_dict, type_='linear'):
    predictions = []
    for index in range(len(X)):
        votes = []
        for c1, c2 in label:
            weight = weight_dict[(c1, c2)]
            bias = bias_dict[(c1, c2)]
            if not type_ == 'linear':
                y = weight.dot(poly_phi(X[index]).T) + bias
            else:
                y = weight.dot(X[index].T) + bias
            if y > 0:
                votes += [c1]
            else:
                votes += [c2]
        predictions += [Counter(votes).most_common()[0][0]]
    return predictions


def svm_plot(support_vectors, X, t, xx, yy, prediction):
    class0_indexes = np.where(t == 0)
    class1_indexes = np.where(t == 1)
    class2_indexes = np.where(t == 2)
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], facecolors='none', edgecolors='k', linewidths=2, label="support vector")
    plt.scatter(X[class0_indexes][:, 0], X[class0_indexes][:, 1], c='r', marker='x', label="class 0")
    plt.scatter(X[class1_indexes][:, 0], X[class1_indexes][:, 1], c='g', marker='*', label="class 1")
    plt.scatter(X[class2_indexes][:, 0], X[class2_indexes][:, 1], c='b', marker='^', label="class 2")
    plt.legend()

    plt.contourf(xx, yy, prediction, alpha=0.3, cmap=plt.cm.coolwarm)


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    space = 0.3
    x_min, x_max = x.min() - space, x.max() + space
    y_min, y_max = y.min() - space, y.max() + space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py#L500
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def PCA_np(features, n=2, svd=0, mean=1, test=0):
    """
    :param features:
    :param n:
    :param svd: 決定要不要使用svd來加快計算的速度
    :return:
    """
    # https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    # https://blog.csdn.net/u012162613/article/details/42177327
    # mean of features
    M = np.mean(features.T, axis=1)
    # center column
    C = features - M
    # eigendecomposition
    if svd:
        # svd will do on the data
        # https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        if mean:
            u, d, v = np.linalg.svd(C, full_matrices=False)  # It's not necessary to compute the full matrix of U or V
        else:
            u, d, v = np.linalg.svd(features, full_matrices=False)
        u, v = svd_flip(u, v)  # 處理正負號(圖就會跟sklearn一致, 沒有就會跟np_eig一致)
        Trans_comp = np.dot(features.T, u[:, :n])
        s = np.diag(d)
        Trans = u[:, :n].dot(s[:n, :n])
        # print(Trans.shape)
        if test:
            return Trans, u[:, :n]
        else:
            return Trans, Trans_comp.T
    else:
        # covariance matrix
        if mean:
            cov = np.cov(C, rowvar=0)  # 求covariance matrix, return ndarray
        else:
            cov = np.cov(features, rowvar=0)
        eigenvalue, eigenvector = np.linalg.eig(np.mat(cov))
        sortEigValue = np.argsort(eigenvalue)  # sort eigenvalue
        topNvalue = sortEigValue[-1:-(n + 1):-1]  # select top n value
        n_eigVect = eigenvector[:, topNvalue]  # select largest n eigenvector
        print(topNvalue.shape, n_eigVect.shape)  # n_eigVect.T is the sklearn pca fit() component_
        # recon = (C*n_eigVect.T) + M  # reconstruct to original data
        Trans = C*n_eigVect  # transform to low dim data (same as the return of sklearn fit_transform())
        # transform matrix to array
        Trans = np.asarray(Trans)
        return Trans, n_eigVect.T


X = pd.read_csv('./data/x_train.csv', header=None).values
y = pd.read_csv('./data/t_train.csv', header=None).iloc[:, 0].values

# X要自行先PCA到2維
X, comp_svd = PCA_np(X, 2, 1)
# normalize
X = (X - X.mean()) / X.std()

# one-versus-one, one-versus-all可以自己調
# 1. linear kernel
svm_model = custom_svm(X, y, vs='ovr')  # 'ovo'跟'ovr'的差異看不太出來
weight_dict, bias_dict = train_svm(X, classes, svm_model.support_, np.abs(svm_model.dual_coef_))
xx, yy = make_meshgrid(X[:, 0], X[:, 1])
prediction = make_prediction(np.column_stack((xx.flatten(), yy.flatten())), classes, weight_dict, bias_dict)
svm_plot(svm_model.support_, X, y, xx, yy, np.array(prediction).reshape(xx.shape))
# plt.savefig('./images/svm_ovo_linearKernel.png')
plt.show()

# 2. polynomial kernel
svm_poly_model = custom_svm(X, y, type_='poly')
weight_dict, bias_dict = train_svm(X, classes, svm_poly_model.support_, np.abs(svm_poly_model.dual_coef_), type_='poly')
# xx, yy = make_meshgrid(X[:, 0], X[:, 1])
prediction = make_prediction(np.column_stack((xx.flatten(), yy.flatten())), classes, weight_dict, bias_dict, 'poly')
svm_plot(svm_poly_model.support_, X, y, xx, yy, np.array(prediction).reshape(xx.shape))
# plt.savefig('./images/svm_ovo_polyKernel.png')
plt.show()

