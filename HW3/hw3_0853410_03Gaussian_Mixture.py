import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd


def gaussian_pdf(x, mean, sd):
    # https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%85%83%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83
    y = 1 / (2 * np.pi) * 1 / np.sqrt(np.linalg.det(sd)) * np.exp(-0.5 * ((x - mean).T.dot(np.linalg.inv(sd))).dot((x - mean)))
    return y


def kmeans_init(K=2):
    mu = data[np.random.choice(len(data), K, replace=False)]
    rnk = np.ones([len(data), K])
    matrixI = np.eye(K)
    return mu, rnk, matrixI


def kmeans_loss(mu, rnk, identity, iter=300):
    for it in range(iter):
        # pixels*"1"*RGB - K*RGB -> pixels*K*RGB -> pixels*K
        dists = np.sum((data[:, None] - mu) ** 2, axis=2)
        # pixels -> pixels*K
        rnk = identity[np.argmin(dists, axis=1)]

        if np.array_equal(rnk, rnk):
            break
        else:
            rnk = rnk

        # 分子： pixels*K*RGB -> K*RGB, 分母：K,1
        mu = np.sum(rnk[:, :, None] * data[:, None], axis=0) / np.sum(rnk, axis=0)[:, None]
    return mu, rnk


def gmm_init(mu, rnk, data, K):
    pi = np.sum(rnk, axis=0) / len(rnk)
    # K*RGB*RGB
    cov = np.array([np.cov(data[np.where(rnk[:, k] == 1)[0]].T) for k in range(K)])
    # pdf: K*pixels
    gaussians = np.array([multivariate_normal.pdf(data, mean=mu[k], cov=cov[k]) * pi[k] for k in range(K)])
    # gaussians = np.array([gaussian_pdf(data, mu[k], cov[k]) * pi[k] for k in range(K)])
    return cov, gaussians


def gmm_EM(gaussians, cov, K):
    # 1. E-steps
    # K*pixels / pixels -> K*pixels -> pixels*K
    gamma = (gaussians / np.sum(gaussians, axis=0)).T
    # 2. M-steps
    Nk = np.sum(gamma, axis=0)
    # K*RGB <- pixels*K*RGB = pixels*K*"1" * pixels*"1"*RGB
    mu = np.sum(gamma[:, :, None] * data[:, None], axis=0) / Nk[:, None]
    # K*RGB*RGB
    for k in range(K):
        # update cov with minivalue to prevent LinAlgError
        cov[k] = (gamma[:, k, None] * (data - mu[k])).T.dot(data - mu[k]) / Nk[k] + 1e-7 * np.eye(depth)
    pi = Nk / len(data)

    return mu, pi, cov, gaussians


def gmm_evaulate(data, it, mu, cov, pi, gaussian, K):
    for k in range(K):
        try:
            gaussian[k] = multivariate_normal.pdf(data, mean=mu[k], cov=cov[k]) * pi[k]
            # gaussian[k] = gaussian_pdf(data, mu[k], cov[k]) * pi[k]
        except np.linalg.linalg.LinAlgError:
            print('singular error at iteration %d' % it)
            mu[k] = np.random.rand(depth)
            temp = np.random.rand(depth, depth)
            cov[k] = temp.dot(temp.T)
            gaussian[k] = multivariate_normal.pdf(data, mean=mu[k], cov=cov[k]) * pi[k]
            # gaussian[k] = gaussian_pdf(data, mu[k], cov[k]) * pi[k]

    return log_likelihood(gaussian), gaussian, cov


def log_likelihood(gaussian):
    return np.sum(np.log(np.sum(gaussian, axis=0)))


def plot_loss(records, K, display=False, save=False, iterMax=100):
    plt.title('Log likelihood of GMM (k=%d)' % K)
    plt.plot([i for i in range(iterMax)], records)  # terminate EM algorithm when the iteration arrives 100
    if save:
        plt.savefig('./images/log_likelihood_' + str(K) + '.png')
    if display:
        plt.show()
    plt.close()


def generate_img(mu, rnk, _type, gaussian, display=False):
    if _type == 'K_means':
        new_data = (mu[np.where(rnk == 1)[1]] * 255).astype(int)
    else:
        new_data = (mu[np.argmax(gaussian, axis=0)] * 255).astype(int)

    disp = Image.fromarray(new_data.reshape(height, width, depth).astype('uint8'))
    if display:
        disp.show(title=_type)
    # disp.save('./images/' + _type + str(k) + '.png')


# use df to print
def print_table(type_, mu, k):
    print("======= K = %d (%s) =======" % (k, type_))
    t1 = [i for i in range(k)]
    t2 = [r for r in (mu[:, 0] * 255).astype(int)]
    t3 = [g for g in (mu[:, 1] * 255).astype(int)]
    t4 = [b for b in (mu[:, 2] * 255).astype(int)]
    d = {type_: t1, 'R': t2, 'G': t3, 'B':t4}
    print(pd.DataFrame(d).to_string(index=False))


img = Image.open('./data/hw3_3.jpeg')
img.load()
data = np.asarray(img, dtype='float')/255
height, width, depth = data.shape
data = np.reshape(data, (-1, depth))  # pixels*RGB  (width*height) = pixels

K_list = [3, 5, 7, 10]
max_iter = 100

for k in K_list:
    muK, rnk, I = kmeans_init(k)
    updated_muK, updated_rnk = kmeans_loss(muK, rnk, I)
    print_table('K_means', updated_muK, k)  # show the table of estimated muK
    generate_img(updated_muK, updated_rnk, 'K_means', None)

    cov, gaussians = gmm_init(updated_muK, updated_rnk, data, k)  # use muK from the K-means model as the means
    records = []
    for it in range(max_iter):
        gmm_mu, pi, cov, gaussians = gmm_EM(gaussians, cov, k)
        loss, gaussians, cov = gmm_evaulate(data, it, gmm_mu, cov, pi, gaussians, k)
        records.append(loss)
    plot_loss(records, k, True)

    generate_img(gmm_mu, None, 'GMM', gaussians, True)
    print_table('GMM', gmm_mu, k)

