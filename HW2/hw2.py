#%%
import pandas as pd
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from collections import Counter
import zipfile
import matplotlib.image as mpimg
# 第一題
# load .mat file
dataMat = scio.loadmat('./dataset/1_data.mat')

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
    # pics[i] = mpimg.imread('./dataset/'+k)

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


def phi(X):
    return np.reshape(X, (len(X), 1))


def cross_entropy(predict, target):
    return -np.sum(predict * np.log(target))


def softmax(n, k, w, X):  # 公式裡的y
    s = np.float64(0.)
    ak = w[k].T.dot(phi(X[n]))
    for j in range(5):
        aj = w[j].T.dot(phi(X[n]))
        s += np.nan_to_num(np.exp(aj - ak))
    s = np.nan_to_num(s)
    return 1. / s


def gradient(w, k, t, X):
    output = np.zeros((len(w[0]), 1))
    for n in range(len(X)):
        scale = softmax(n, k, w, X) - t[:, k][n]  # Ynk - Tnk
        output += scale * phi(X[n])
    return output


def hessian(w, k, X):
    output = np.zeros((len(w[0]), len(w[0])))
    for n in range(len(X)):
        scale = softmax(n, k, w, X) * (1 - softmax(n, k, w, X))
        output += scale * (phi(X[n]).dot(phi(X[n]).T))
    return output


def error(w,t,X):
    s = np.float64(0.)
    for n in range(len(X)):
        for k in range(5):
            if t[:,k][n] != 0.:
                s += np.nan_to_num(np.log(softmax(n,k,w,X)))
    return -1*s


def classify(w,x):
    softmaxes = []
    for k in range(3):
        s = np.float64(0.)
        ak = w[k].T.dot(phi(x))
        for j in range(3):
            aj = w[j].T.dot(phi(x))
            s += np.nan_to_num(np.exp(aj - ak))
        softmaxes += [1./s]
    return softmaxes.index(max(softmaxes))


def softmax_result(z):
    z -= np.max(z)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T


# 1. GD
classes = 5
w_1 = np.zeros([feature.shape[1], classes])
losses = []
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
    loss = (-1/ m) * cross_entropy(y1, prob)
    losses.append(loss)
    grad = (-1/ m) * np.dot(x1.T, (y1 - prob))
    w_1 -= lr*grad

# plot GD train loss
plt.plot(losses)
plt.xlabel('Number Epochs')
plt.ylabel('Loss')
plt.show()

# predict ohe class
pred_score = softmax_result(np.dot(X_test, w_1))
pred = np.where(pred_score > 0.5, 1, 0)
# test acc
# all with axis equal 1 will compare whether each row matches the answer
print((pred == y_test).all(axis=1).mean())
# test loss
print((-1/25)*cross_entropy(y_test, pred_score))


# todo: 畫PCA
def PCA_np(features, n=2):
    # https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    # https://blog.csdn.net/u012162613/article/details/42177327
    # mean of features
    M = np.mean(features.T, axis=1)
    # center column
    C = features - M
    t0 = datetime.datetime.now()
    # covariance matrix
    cov = np.cov(C, rowvar=0)  # 求covariance matrix, return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    print(datetime.datetime.now() - t0)
    # eigendecomposition
    eigenvalue, eigenvector = np.linalg.eig(np.mat(cov))
    print(datetime.datetime.now() - t0)
    sortEigValue = np.argsort(eigenvalue)  # sort eigenvalue
    topNvalue = sortEigValue[-1:-(n + 1):-1]  # select top n value
    n_eigVect = eigenvector[:, topNvalue]  # select largest n eigenvector
    print(topNvalue.shape, n_eigVect[0].shape, n_eigVect.shape)  # n_eigVect.T is the sklearn pca fit() component_
    # recon = (C*n_eigVect.T) + M  # reconstruct to original data
    return C*n_eigVect, n_eigVect.T  # transform to low dim data (same as the return of sklearn fit_transform())


import datetime
t1 = datetime.datetime.now()
# comp is a matrix
pca55, comp = PCA_np(feature, 5)
# convert complex type to float
# [n.real for n in np.asarray(comp)]
print(datetime.datetime.now()-t1)
# https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction
from PIL import Image
from sklearn.decomposition import PCA
pca = PCA(5)
pca5 = pca.fit(feature)  # pca5.components_ 為(5, 92)應該是eigenvector
print(pca5.components_[0].shape)
print(np.equal(np.asarray([n.real for n in np.asarray(comp)]), pca5.components_))
img = Image.fromarray(pca5.components_[0].reshape(112, 92))
img = Image.fromarray(np.asarray([n.real for n in np.asarray(comp)[0]]).reshape(112, 92))
img.save('./eigenface3.png')
img.show()
plt.imshow(np.asarray([n.real for n in np.asarray(comp)[0]]).reshape(112, 92))
plt.imshow(np.asarray([n.real for n in np.asarray(comp)[0]]).reshape(112, 92), cmap='gray')
plt.show()

# todo: train起來
w = np.zeros((classes, len(phi(feature[0])), 1))
cee = []
acc = []
# epoch
for ep in range(20):
    e = error(w, target, feature)
    cee += [np.reshape(e, 1)]
    for k in range(classes):
        # 2. 牛頓法
        w[k] = w[k] - np.linalg.inv(hessian(w, k, feature)).dot(gradient(w, k, target, feature))
    break


A = np.array([[1, 2], [3, 4], [5, 6]])
PCA_np(A, 1)


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

# todo: 測試分開train, test做normalize
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
    # fixme: 用原始照片, 每次帶一張做pca
    # convert np matrix to array
    pca_data = np.asarray(PCA_np(pokemon_norm.values, d))
    for k in range(1, 11):
        predictions = []
        #  compare each test sample with 120 training samples
        for j in range(38):
            j = 120 + j
            predictions.append(knn(pca_data[j], pca_data[:120], target[:120], k))
        # 計算Acc
        acc.append(np.count_nonzero(predictions == target[120:]) / len(target[120:]))

    plot_knn(acc, range(1, 11))

#%%
