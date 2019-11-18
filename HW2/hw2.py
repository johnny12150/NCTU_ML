#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Counter
# 第一題

#%%
# 第二題
def PCA_np(features, n=2):
    # https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    # mean of features
    M = np.mean(features.T, axis=1)
    # center column
    C = features - M
    # covariance matrix
    cov = np.cov(C.T)
    # eigendecomposition
    eigenvalue, eigenvector = np.linalg.eig(cov)
    mat = np.matrix(eigenvector[:n])
    print(eigenvector[:][:1].shape, C.shape)
    print(eigenvector[:][:1], eigenvector[:1], eigenvector)
    return eigenvector[:n].T.dot(C.T)
    # if len(eigenvector[:n]) == len(features[0]):
    #     # project data
    #     return eigenvector[:].T.dot(C.T)
    # else:
    #     return np.matrix(mat.T)


# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca2(dataMat, n):
    # https://blog.csdn.net/u012162613/article/details/42177327
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat
    # return lowDDataMat, reconMat

A = np.array([[1, 2], [3, 4], [5, 6]])
PCA_np(A, 1).T # same as sklearn
pca2(A, 2)

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
    # feature有哪些種類
    # print(pokemon[col].value_counts().index.astype('str').tolist())
    # 做one hot
    pokemon_ohe[col], uni = pokemon[col].factorize()
    uniques.append(uni.tolist())

pokemon_norm = pokemon_ohe.copy()
# normalize
pokemon_norm = (pokemon_norm - pokemon_norm.mean()) / pokemon_norm.std()

pokemon_np = pokemon_ohe.values
# np style
pokemon_np = (pokemon_np - np.mean(pokemon_np, 0)) / np.std(pokemon_np, 0)

target = pokemon_norm['Type 1'].copy()
pokemon_norm = pokemon_norm.drop(['Type 1'], axis=1)
acc = []
for k in range(1, 11):
    predictions = []
    #  compare each test sample with 120 training samples
    for j in range(38):
        j = 120+j
        predictions.append(knn(pokemon_norm.iloc[j, :], pokemon_norm[:120], target[:120], k))
    # 計算Acc
    acc.append(np.count_nonzero(predictions == target[120:]) / len(target[120:]))

