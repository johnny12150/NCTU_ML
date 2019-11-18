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
    # print(eigenvalue, eigenvector)
    mat = np.matrix(eigenvector[:n])
    if len(eigenvector[:n]) == len(features[0]):
        # project data
        return eigenvector[:].T.dot(C.T)
    else:
        return np.matrix(mat.T)


def PCA_numpy(data, n_components=2):
    # 1nd step is to find covarience matrix
    data_vector = []
    for i in range(data.shape[1]):
        data_vector.append(data[:, i])

    cov_matrix = np.cov(data_vector)

    # 2rd step is to compute eigen vectors and eigne values
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    eig_values = np.reshape(eig_values, (len(cov_matrix), 1))

    # Make pairs
    eig_pairs = []
    for i in range(len(eig_values)):
        eig_pairs.append([np.abs(eig_values[i]), eig_vectors[:, i]])

    eig_pairs.sort()
    eig_pairs.reverse()

    # This PCA is only for 2 components
    reduced_data = np.hstack((eig_pairs[0][1].reshape(len(eig_pairs[0][1]), 1), eig_pairs[1][1].reshape(len(eig_pairs[0][1]), 1)))

    return data.dot(reduced_data)

A = np.array([[1, 2], [3, 4], [5, 6]])
PCA_np(A)
# PCA_numpy(A, 1)

from sklearn.decomposition import PCA
pca = PCA(2)
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

