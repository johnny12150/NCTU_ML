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
archive.extractall('./dataset/')
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

# todo: random select train


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


# todo 1. GD
# fixme: picture flatten, target ohe
feature = pics_norm.reshape(50, -1)
target = pd.get_dummies(labels).values
# target.columns = list(range(5))
classes = 5
w = np.zeros((classes, len(phi(feature[0])), 1))
cee = []
acc = []
lr = 0.1
# epoch
for ep in range(20):
    # x = feature.dot(w)
    # prediction
    p = error(w, target, feature)
    for k in range(classes):
        # fixme: w is too small
        w -= lr*gradient(w, k, target, feature)
    break

# todo: train起來
w = np.zeros((classes, len(phi(feature[0])), 1))
cee = []
acc = []
# epoch
for ep in range(20):
    e = error(w,target, feature)
    acc += [accuracy(w,target,feature)]
    cee += [np.reshape(e, 1)]
    for k in range(classes):
        # 2. 牛頓法
        w[k] = w[k] - np.linalg.inv(hessian(w,k,feature)).dot(gradient(w,k,target,feature))


def PCA_np(features, n=2):
    # https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    # https://blog.csdn.net/u012162613/article/details/42177327
    # mean of features
    M = np.mean(features.T, axis=1)
    # center column
    C = features - M
    # covariance matrix
    cov = np.cov(C, rowvar=0) # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    # eigendecomposition
    eigenvalue, eigenvector = np.linalg.eig(np.mat(cov))
    sortEigValue = np.argsort(eigenvalue)  # sort eigenvalue
    topNvalue = sortEigValue[-1:-(n + 1):-1]  # select top n value
    n_eigVect = eigenvector[:, topNvalue]  # select largest n eigenvector
    # recon = (C*n_eigVect.T) + M  # reconstruct to original data
    return C*n_eigVect  # transform to low dim data


A = np.array([[1, 2], [3, 4], [5, 6]])
PCA_np(A, 1)

from sklearn.decomposition import PCA
pca = PCA(1)
pca.fit_transform(A)


#%%
# 第三題
pokemon = pd.read_csv('./dataset/Pokemon.csv')


def euclidean_distance(d1, d2):
    """
    計算歐基里德距離
    :param d1: train
    :param d2: test
    :return:
    """
    distance = (d1 - d2)**2
    return np.sqrt(np.sum(distance, 1))


def knn(train, test, target, k):
    distances = euclidean_distance(train, test)
    idx = sorted(range(len(distances)), key=lambda i: distances[i])[0:k]
    return Counter(target[idx]).most_common(1)[0][0]


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

target = pokemon_norm['Type 1'].copy()
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
