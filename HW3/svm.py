import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from collections import Counter


# fixme: 自己重寫
class Kernel:
    def __init__(self, _type='linear'):
        self._type = _type
        self.phi_x = self.linear_phi if _type == 'linear' else self.poly_phi

    def linear_phi(self, x):
        return x

    def poly_phi(self, x):
        if len(x.shape) == 1:
            return np.vstack((x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2)).T  # shape: 1*3
        else:
            return np.vstack((x[:, 0] ** 2, np.sqrt(2) * x[:, 0] * x[:, 1], x[:, 1] ** 2)).T

    def kernel_function(self, xn, xm):
        return np.dot(self.phi_x(xn), self.phi_x(xm).T)


class SVM:
    def __init__(self, _type='linear', C=1):
        self._type = _type
        self.kernel = Kernel(_type)
        self.class_label = [(0, 1), (0, 2), (1, 2)]
        self.C = C
        self.coef = None
        self.sv_index = None

    def fit(self, X, y):
        if self._type == 'linear':
            clf = SVC(kernel='linear', C=self.C, decision_function_shape='ovo')
        else:
            clf = SVC(kernel='poly', C=self.C, degree=2, decision_function_shape='ovo')
        clf.fit(X, y)
        # dual_coef_[i] = labels[i] * alphas[i] where labels[i] is either -1 or +1 and alphas[i] are always positive
        self.coef = np.abs(clf.dual_coef_)
        self.sv_index = clf.support_

    def prepare_parameter_for_classifiers(self, X):
        size = 100
        # target
        target_dict = {}
        target_dict[(0, 1)] = np.concatenate((np.ones(size), np.full([size], -1), np.zeros(size)))
        target_dict[(0, 2)] = np.concatenate((np.ones(size), np.zeros(size), np.full([size], -1)))
        target_dict[(1, 2)] = np.concatenate((np.zeros(size), np.ones(size), np.full([size], -1)))

        # multiplier
        multiplier = np.zeros([len(X), 2])
        multiplier[self.sv_index] = self.coef.T

        multiplier_dict = {}
        multiplier_dict[(0, 1)] = np.concatenate((multiplier[:size*2, 0], np.zeros(size)))
        multiplier_dict[(0, 2)] = np.concatenate((multiplier[:size, 1], np.zeros(size), multiplier[size*2:, 0]))
        multiplier_dict[(1, 2)] = np.concatenate((np.zeros(size), multiplier[size:, 1]))
        return target_dict, multiplier_dict

    def get_w_b(self, a, t, x):
        at = a * t  # PRML 7.29
        w = at.dot(self.kernel.phi_x(x))
        # PRML 7.37
        M_indexes = np.where(((a > 0) & (a < self.C)))[0]
        S_indexes = np.nonzero(a)[0]
        Nm = len(M_indexes)

        if Nm == 0:
            b = -1  # ???
        else:
            #   b = np.mean(t[M_indexes] - np.linalg.multi_dot([at[S_indexes], x[S_indexes], x[M_indexes].T]))
            b = np.mean(t[M_indexes] - at[S_indexes].dot(self.kernel.kernel_function(x[M_indexes], x[S_indexes]).T))

        return w, b

    def train(self, X, t):
        target_dict, multiplier_dict = self.prepare_parameter_for_classifiers(X)
        weight_dict = {}
        bias_dict = {}

        for c1, c2 in self.class_label:
            weight, bias = self.get_w_b(multiplier_dict[(c1, c2)], target_dict[(c1, c2)], X)
            weight_dict[(c1, c2)] = weight
            bias_dict[(c1, c2)] = bias
        return weight_dict, bias_dict

    def predict(self, X, weight_dict, bias_dict):
        prediction = []
        for index in range(len(X)):
            votes = []
            for c1, c2 in self.class_label:
                weight = weight_dict[(c1, c2)]
                bias = bias_dict[(c1, c2)]
                y = weight.dot(self.kernel.phi_x(X[index]).T) + bias
                if y > 0:
                    votes += [c1]
                else:
                    votes += [c2]
            prediction += [Counter(votes).most_common()[0][0]]
        return prediction

    def plot(self, X, t, xx, yy, prediction):
        class0_indexes = np.where(t == 0)
        class1_indexes = np.where(t == 1)
        class2_indexes = np.where(t == 2)
        plt.scatter(X[self.sv_index, 0], X[self.sv_index, 1], facecolors='none', edgecolors='k', linewidths=2,
                    label="support vector")
        plt.scatter(X[class0_indexes][:, 0], X[class0_indexes][:, 1], c='r', marker='x', label="class 0")
        plt.scatter(X[class1_indexes][:, 0], X[class1_indexes][:, 1], c='g', marker='*', label="class 1")
        plt.scatter(X[class2_indexes][:, 0], X[class2_indexes][:, 1], c='b', marker='^', label="class 2")
        plt.legend()

        plt.contourf(xx, yy, prediction, alpha=0.3, cmap=plt.cm.coolwarm)

    def make_meshgrid(self, x, y, h=0.02):
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

# todo: X要自行先PCA到2維
X, comp_svd = PCA_np(X, 2, 1)
# normalize
X = (X - X.mean()) / X.std()

# fixme: one-versus-rest實作
# 以下都是用 one-versus-one
# 1. linear kernel
svm_linear = SVM()
# this would cost a long time
svm_linear.fit(X, y)
weight_dict, bias_dict = svm_linear.train(X, y)
xx, yy = svm_linear.make_meshgrid(X[:, 0], X[:, 1])
prediction = svm_linear.predict(np.column_stack((xx.flatten(), yy.flatten())), weight_dict, bias_dict)
svm_linear.plot(X, y, xx, yy, np.array(prediction).reshape(xx.shape))
plt.savefig('./images/svm_ovo_linearKernel.png')
plt.show()

# 2. polynomial kernel
svm_poly = SVM(_type='poly')
svm_poly.fit(X, y)
weight_dict, bias_dict = svm_poly.train(X, y)
xx, yy = svm_poly.make_meshgrid(X[:, 0], X[:, 1])
prediction = svm_poly.predict(np.column_stack((xx.flatten(), yy.flatten())), weight_dict, bias_dict)
svm_poly.plot(X, y, xx, yy, np.array(prediction).reshape(xx.shape))
plt.savefig('./images/svm_ovo_polyKernel.png')
plt.show()

