#%%
import pandas as pd
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import zipfile
import math
# 第一題
# load .mat file
dataMat = scio.loadmat('./dataset/1_data.mat')
x = dataMat['x']
t = dataMat['t']

M = 3


def phi(x, m, trans=1):
    X = []
    for i in range(m):
        # X += [sigmoid_basis(x,i, m)]
        X.append(sigmoid_basis(x, i, m))
    if trans:
        return np.array(X).reshape(-1, m)
    else:
        return np.array(X)


def sigmoid_basis(x, j, m, sigma=0.1):
    muj = (2 * j) / m
    a = (x - muj) / sigma
    return 1 / (1 + np.exp(-a))


def predictive_dist(x, M, mN, SN, beta=1):
    phiX = phi(x, M, 0).T
    mean = phiX.dot(mN)
    covX = 1 / beta + np.sum(phiX.dot(SN).dot(phiX.T), axis=1)
    std = np.sqrt(covX)
    return mean, std


def predict(w, x):
    return w.dot(x.T)


def gaussian_pdf(x, mean, sd):
    # https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%85%83%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83
    y = 1 / (2 * np.pi) * 1 / np.sqrt(np.linalg.det(sd)) * np.exp( -0.5 * ((x - mean).T.dot(np.linalg.inv(sd))).dot((x - mean)))
    return y

s0_inv = (10 ** -6) * np.identity(M)  # S0
m0 = 0
beta = 1
MNs = []
SNs = []
data_len = [5, 10, 30, 80]

# 大PHI矩陣(裡面有許多小phi)
PHI = phi(x[0], M)
sn_inv = s0_inv + beta * PHI.T.dot(PHI)
sn = np.linalg.inv(sn_inv)
mn = sn.dot(beta * np.dot(PHI.T, t[0]))
# 每次多拿到一筆資料來更新M, S
for i in range(1, x.shape[0]):
    # 存特定data筆數的mean跟std
    if i in data_len:
        MNs.append(mn)
        SNs.append(sn)
        # 第一種圖(Fig. 3.9)
        # plot data point
        plt.scatter(x[:i], t[:i], facecolor="none", edgecolor="b", label="training data")
        plt.legend()
        # sample five curve
        w_sampled = np.random.multivariate_normal(mn, sn, size=5)  # same as scipy multivariate_normal
        # sortX = np.array(sorted(x))
        sortX = np.linspace(0, 2, 50)
        pred = predict(w_sampled, phi(sortX, M, 0).T)
        # plot five curve that we just sampled
        plt.title('data size %d' % i)
        for j in range(5):
            plt.plot(sortX, pred[j], '-r')
        plt.show()

        # 2.第二種圖(Fig. 3.8)predictive distribution
        mean, std = predictive_dist(sortX, M, mn, sn)
        plt.title('data size %d' % i)
        plt.scatter(x[:i],t[:i], facecolor="none", edgecolor="b", label="training data")
        plt.plot(sortX, mean, 'r', label='mean')
        plt.fill_between(sortX.reshape(len(sortX)), mean - std, mean + std, alpha=0.5, color='orange', label='std')
        plt.legend()
        plt.show()

        # 3. Fig. 3.7
        plt.title('data size %d' % i)
        w0, w1 = np.meshgrid(np.linspace(0, 5, 100), np.linspace(-2, 3, 100))
        w_combined = np.array([w0, w1]).transpose(1, 2, 0)
        N_density = np.empty((100, 100))
        for f in range(N_density.shape[0]):
            for g in range(N_density.shape[1]):
                # select weight
                N_density[f, g] = gaussian_pdf(w_combined[f, g], mn_old[:2], np.linalg.inv(sn_inv_old)[:2, :2])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.contourf(w0[0], w1[:, 0], N_density)
        plt.show()

    # update M, S
    PHI = np.vstack((PHI, phi(x[i], M)))
    mn_old = mn
    sn_inv_old = sn_inv
    sn_inv = sn_inv + beta * PHI.T.dot(PHI)  # 計算Sn，且beta = 1
    sn = np.linalg.inv(sn_inv)
    # sn_inv跟mn要帶舊的
    mn = sn.dot(sn_inv_old.dot(mn) + beta * PHI.T.dot(t[:i + 1]).reshape(-1, ))

#%%
# 第二題
# https://stackoverflow.com/questions/46588075/read-all-files-in-zip-archive-in-python
# https://stackoverflow.com/questions/19371860/python-open-file-from-zip-without-temporary-extracting-it
# load zip
archive = zipfile.ZipFile('./dataset/Faces.zip', 'r')
# export unzipped files
# archive.extractall('./dataset/')
# make dict for all pictures
files = {name: archive.read(name) for name in archive.namelist() if name.endswith('.pgm')}
# fixme: 如果解壓縮失敗，能夠自動讀取已經解壓縮的
pic_height = 112
pic_width = 92
# load pics ti np array
pics = np.zeros((50, pic_height, pic_width))
labels = np.repeat(range(1, 6), 10)
pic_header = len(b'P5\n92 112\n255\n')
for i, k in enumerate(files):
    # read pgm, https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    pics[i] = np.frombuffer(files[k], dtype='u1', count=pic_width*pic_height, offset=pic_header).reshape((pic_height, pic_width))

# normalize pics
pics_norm = pics/ 255
feature = pics_norm.reshape(50, -1)  # picture flatten
target = pd.get_dummies(labels).values  # ohe target

idxs = []
t_idxs = []
# random select train
for i in range(5):
    # random select five for training in each subject
    idx = np.random.choice(np.arange(10), 5, replace=False) + i*10
    idxs.extend(idx)
    # the rest is test
    t_idx = list(set(np.arange(i*10, 10 + i*10)) - set(idx))
    t_idxs.extend(t_idx)

X_train = feature[idxs]
y_train = target[idxs]
X_test = feature[t_idxs]
y_test = target[t_idxs]

# random shuffle
randomize = np.arange(25)
np.random.shuffle(randomize)
X_train = X_train[randomize]
y_train = y_train[randomize]
X_test = X_test[randomize]
y_test = y_test[randomize]


def phi(x):
    return x.reshape(len(x), 1)


def cross_entropy(target, predict):
    return -np.sum(target * np.log(predict))


def compute_y(n, k, w, X):  # 公式裡的y
    s = np.float64(0.)
    ak = w[k].T.dot(phi(X[n]))
    # target classes
    for j in range(5):
        aj = w[j].T.dot(phi(X[n]))
        s += np.nan_to_num(np.exp(aj - ak))
    s = np.nan_to_num(s)
    return 1. / s


def gradient(w, k, t, X):
    output = np.zeros((len(w[0]), 1))
    for n in range(len(X)):
        scale = compute_y(n, k, w, X) - t[:, k][n]  # Ynk - Tnk
        output += scale * phi(X[n])
    return output


# 對error二次微分
def hessian(w, k, X):
    output = np.zeros((len(w[0]), len(w[0])))
    for n in range(len(X)):
        scale = compute_y(n, k, w, X) * (1 - compute_y(n, k, w, X))
        output += scale * (phi(X[n]).dot(phi(X[n]).T))
    return output


def error(w, t, X):
    s = np.float64(0.)
    for n in range(len(X)):
        for k in range(5):
            if t[:, k][n] != 0.:
                s += np.nan_to_num(np.log(compute_y(n, k, w, X)))
    return -1*s


def softmax_result(z):
    z -= np.max(z)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T


# 1. GD
classes = 5
w_1 = np.zeros([feature.shape[1], classes])
tr_accuracy = []
tr_losses = []
te_accuracy = []
te_losses = []
lr = 1e-3
x1 = X_train
y1 = y_train
# epoch
for ep in range(1000):
    # https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
    # value of w * x
    score = np.dot(x1, w_1)
    prob = softmax_result(score)
    m = x1.shape[0]
    # cross entropy
    loss = (1/ m) * cross_entropy(y1, prob)
    # record acc and loss
    tr_losses.append(loss)
    pred_score = softmax_result(np.dot(X_train, w_1))
    pred = np.where(pred_score > 0.5, 1, 0)
    tr_accuracy.append((pred == y_train).all(axis=1).mean())
    # test on epoch
    pred_score = softmax_result(np.dot(X_test, w_1))
    pred = np.where(pred_score > 0.5, 1, 0)
    te_losses.append((1/m)*cross_entropy(y_test, pred_score))
    te_accuracy.append((pred == y_test).all(axis=1).mean())
    # update weight
    grad = (-1/ m) * np.dot(x1.T, (y1 - prob))
    w_1 -= lr*grad


def plot_loss_acc(record, x_axis, y_axis, tl='Train'):
    plt.title(tl + '  ' + y_axis)
    plt.plot(record)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(range(1, len(record)+1))
    plt.show()


# plot GD train loss
plot_loss_acc(tr_losses, 'Number Epochs', 'Loss')
# train acc
plot_loss_acc(tr_accuracy, 'Number Epochs', 'Accuracy')

# test acc
plot_loss_acc(te_losses, 'Number Epochs', 'Loss', 'Test')
# train acc
plot_loss_acc(te_accuracy, 'Number Epochs', 'Accuracy', 'Test')

# predict ohe class
pred_score = softmax_result(np.dot(X_test, w_1))
pred = np.where(pred_score > 0.5, 1, 0)
# all with axis= 1 will compare whether each row matches the answer
print((pred == y_test).all(axis=1).mean())
# test loss
print((1/25)*cross_entropy(y_test, pred_score))


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


# comp is a matrix
pca_svs, comp_svd = PCA_np(feature, 5, 1)
# this one is too slow
# pca_eig, comp = PCA_np(feature, 5)  # this one returns matrix
# https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction


def plot_eigenface(eig, color='gray'):
    """
    https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction
    :param eig:
    :param color: 'gray' or none
    """
    if color:
        plt.imshow(eig, cmap=color)
    else:
        plt.imshow(eig)
    # remove coordination
    plt.xticks(())
    plt.yticks(())
    plt.show()


for i in range(comp_svd.shape[0]):
    # convert complex type to float, [n.real for n in np.asarray(comp)
    # plot_eigenface(np.asarray([n.real for n in np.asarray(comp)[0]]).reshape(112, 92))
    plot_eigenface(comp_svd[i].reshape(112, 92))  # for svd


def predicts(w, x, classes):
    softmaxes = []
    for k in range(classes):
        s = np.float64(0.)
        ak = w[k].T.dot(phi(x))
        for j in range(classes):
            aj = w[j].T.dot(phi(x))
            s += np.nan_to_num(np.exp(aj - ak))
        softmaxes += [1./s]
    return np.where(np.array(softmaxes).reshape(-1) > 1/ classes, 1, 0)
    # 回傳預測的class
    # return softmaxes.index(max(softmaxes))


dim = [2, 5, 10]
for d in dim:
    err = []
    acc = []
    t_err = []
    t_acc = []
    pca_feature, pca_com = PCA_np(X_train, d, 1)
    w = np.zeros((classes, len(phi(pca_feature[0])), 1))
    # epoch
    for ep in range(10):
        e = error(w, y_train, pca_feature)
        err += [np.reshape(e, 1)]
        for k in range(classes):
            # 2. 牛頓法
            w[k] = w[k] - np.linalg.inv(hessian(w, k, pca_feature)).dot(gradient(w, k, y_train, pca_feature))
        # make predictions base the training weight
        prediction = []
        for n in pca_feature:
            prediction.append(predicts(w, n, classes))
        # use ohe label
        acc.append((np.array(prediction) == y_train).all(axis=1).mean())

        # testing
        e = error(w, y_test, pca_feature)
        t_err += [np.reshape(e, 1)]
        pca_feature, pca_com = PCA_np(X_test, d, 1)
        prediction = []
        for n in pca_feature:
            prediction.append(predicts(w, n, classes))
        t_acc.append((np.array(prediction) == y_test).all(axis=1).mean())

    plot_loss_acc(err, 'Number Epochs', 'Loss', 'Dim = '+str(d)+' Training')
    plot_loss_acc(acc, 'Number Epochs', ' Accuracy', 'Dim = '+str(d)+' Training')

    plot_loss_acc(t_err, 'Number Epochs', 'Loss', 'Dim = '+str(d)+' Testing')
    plot_loss_acc(t_acc, 'Number Epochs', 'Accuracy', 'Dim = '+str(d)+' Testing')


#%%
# 第三題
pokemon = pd.read_csv('./dataset/Pokemon.csv', index_col=0)


def euclidean_distance(d1, d2):
    """
    計算歐基里德距離
    :param d1: train
    :param d2: test
    :return:
    """
    distance = (d2 - d1)**2
    return np.sqrt(np.sum(distance, 1))


def knn(test, train, target, k):
    distances = euclidean_distance(train, test)
    # 照距離排序，找最近的K個的index
    idx = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    # 替代方案
    # idx = np.argsort(distances)[:k]
    # 各類別有幾個，挑最多個那個class
    un, cn = np.unique(target[idx], return_counts=True)
    # 替代方案
    # np.bincount(target[idx]).argmax()
    # most common與np的比較
    # if not np.equal(un[cn.argmax()], Counter(target[idx]).most_common(1)[0][0]):
    #     print(Counter(target[idx]))
    #     print(Counter(target[idx]).most_common(1)[0][0], un[cn.argmax()])
    return un[cn.argmax()]


def plot_knn(acc, k_range):
    plt.plot(k_range, acc)
    plt.xticks(k_range)
    plt.ylabel('Accuracy')
    plt.show()


# 找出那些column需要轉
categorical_col = []
for col in pokemon:
    if pokemon[col].dtype == 'O':
        categorical_col.append(col)
    elif pokemon[col].dtype == 'bool':
        categorical_col.append(col)

pokemon_ohe = pokemon.copy()
uniques = []
for col in categorical_col:
    # 用factorize做one hot
    pokemon_ohe[col], uni = pokemon[col].factorize()
    uniques.append(uni.tolist())

pokemon_norm = pokemon_ohe.copy()
# normalize
pokemon_norm = (pokemon_norm - pokemon_norm.mean()) / pokemon_norm.std()

target_ohe = pokemon_norm['Type 1'].copy()
target = pokemon_ohe['Type 1'].values
pokemon_norm = pokemon_norm.drop(['Type 1'], axis=1)
acc = []
for k in range(1, 11):
    predictions = []
    #  compare each test sample with 120 training samples
    for j in range(38):
        j = 120+j
        predictions.append(knn(pokemon_norm.iloc[j, :], pokemon_norm[:120], target[:120], k))
    # 1. 計算Acc
    acc.append(np.count_nonzero(predictions == target[120:]) / len(target[120:]))

plot_knn(acc, range(1, 11))

# 2. PCA到7, 6, 5維
dim = [7, 6, 5]
for d in dim:
    acc = []
    # 用 train取得 eigenvector, test用 train的作 transform
    # pca_data, pca_vec = PCA_np(pokemon_norm.values[:120], d, 0, 0)
    # pca_test = np.asarray(pokemon_norm.values[120:]*pca_vec.T)
    # below will return the same result as sklearn
    pca_data_s, pca_vec_s = PCA_np(pokemon_ohe.drop(['Type 1'], axis=1).values[:120], d, 0, 1)
    # 理論上test data要自己 normalize
    pca_test_s = np.asarray(pokemon_norm.values[120:] * pca_vec_s.T)

    # svd版, 畫出來的合理一些
    pca_d, pca_u = PCA_np(pokemon_norm.values[:120], d, 1, 0)
    pca_data = np.dot(pokemon_norm.values[:120], pca_u.T)
    pca_test = np.dot(pokemon_norm.values[120:], pca_u.T)

    for k in range(1, 11):
        predictions = []
        #  compare each test sample with 120 training samples
        for j in range(38):
            # predictions.append(knn(pca_test_s[j], pca_data_s[:120], target[:120], k))
            predictions.append(knn(pca_test[j], pca_data[:120], target[:120], k))
        # 計算Acc
        acc.append(np.count_nonzero(predictions == target[120:]) / len(target[120:]))

    plot_knn(acc, range(1, 11))

#%%
