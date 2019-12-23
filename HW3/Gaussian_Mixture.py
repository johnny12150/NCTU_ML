import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# fixme: 自己重寫
class K_means:
    def __init__(self, K=2):
        self.K = K
        self.max_iteration = 300
        self.eye = np.eye(K)

    def initialize(self, data):
        self.mu = data[np.random.choice(len(data), self.K, replace=False)]
        self.rnk = np.ones([len(data), self.K])

    def minimize_J(self, data):
        for it in range(self.max_iteration):
            # pixels*"1"*RGB - K*RGB -> pixels*K*RGB -> pixels*K
            dists = np.sum((data[:, None] - self.mu) ** 2, axis=2)
            # pixels -> pixels*K
            rnk = self.eye[np.argmin(dists, axis=1)]

            if np.array_equal(rnk, self.rnk):
                break
            else:
                self.rnk = rnk

            # 分子： pixels*K*RGB -> K*RGB, 分母：K,1
            self.mu = np.sum(rnk[:, :, None] * data[:, None], axis=0) / np.sum(rnk, axis=0)[:, None]


class GMM:
    def __init__(self, K, max_iteration=100):
        self.K = K
        self.max_iteration = max_iteration
        self.likelihood_records = []

    def initialize(self, k_means_rnk, k_means_mu, data):
        self.pi = np.sum(k_means_rnk, axis=0) / len(k_means_rnk)
        # K*RGB*RGB
        self.cov = np.array([np.cov(data[np.where(k_means_rnk[:, k] == 1)[0]].T) for k in range(self.K)])
        # pdf: K*pixels
        self.gaussians = np.array(
            [multivariate_normal.pdf(data, mean=k_means_mu[k], cov=self.cov[k]) * self.pi[k] for k in range(self.K)])

    def E_step(self):
        # K*pixels / pixels -> K*pixels -> pixels*K
        self.gamma = (self.gaussians / np.sum(self.gaussians, axis=0)).T

    def M_step(self, data):
        self.Nk = np.sum(self.gamma, axis=0)
        # K*RGB <- pixels*K*RGB = pixels*K*"1" * pixels*"1"*RGB
        # self.mu = self.gamma.T.dot(self.data)
        self.mu = np.sum(self.gamma[:, :, None] * data[:, None], axis=0) / self.Nk[:, None]
        # K*RGB*RGB
        for k in range(self.K):
            # update cov with minivalue to prevent LinAlgError
            self.cov[k] = (self.gamma[:, k, None] * (data - self.mu[k])).T.dot(data - self.mu[k]) / self.Nk[
                k] + 1e-7 * np.eye(depth)
        self.pi = self.Nk / len(data)

    def evaulate(self, data, it):
        for k in range(self.K):
            try:
                self.gaussians[k] = multivariate_normal.pdf(data, mean=self.mu[k], cov=self.cov[k]) * self.pi[k]
            except np.linalg.linalg.LinAlgError:
                print('singular error at iteration %d' % it)
                self.mu[k] = np.random.rand(depth)
                temp = np.random.rand(depth, depth)
                self.cov[k] = temp.dot(temp.T)
                self.gaussians[k] = multivariate_normal.pdf(data, mean=self.mu[k], cov=self.cov[k]) * self.pi[k]

        self.likelihood_records.append(self.log_likelihood())

    def log_likelihood(self):
        return np.sum(np.log(np.sum(self.gaussians, axis=0)))

    def EM(self, data):
        for it in range(self.max_iteration):
            self.E_step()
            self.M_step(data)
            self.evaulate(data, it)

    def plot_likelihood_log(self):
        plt.title('Log likelihood of GMM (k=%d)' % self.K)
        plt.plot([i for i in range(100)], self.likelihood_records)  # terminate EM algorithm when the iteration arrives 100
        plt.savefig('./images/log_likelihood_' + str(self.K) + '.png')
        plt.close()
        # plt.show()


# def print_RGB_table(model, _type):
#     tb = pt.PrettyTable()
#     tb.add_column(_type, [k for k in range(model.K)])
#     tb.add_column('R', [r for r in (model.mu[:, 0] * 255).astype(int)])
#     tb.add_column('G', [g for g in (model.mu[:, 1] * 255).astype(int)])
#     tb.add_column('B', [b for b in (model.mu[:, 2] * 255).astype(int)])
#     print("======= K = %d (%s) =======" % (k, _type))
#     print(tb)
#     print()


def generate_img(model, _type):
    if _type == 'K_means':
        new_data = (model.mu[np.where(k_means.rnk == 1)[1]] * 255).astype(int)
    else:
        new_data = (model.mu[np.argmax(model.gaussians, axis=0)] * 255).astype(int)

    disp = Image.fromarray(new_data.reshape(height, width, depth).astype('uint8'))
    # disp.show(title=_type)
    disp.save('./images/' + _type + str(k) + '.png')

img = Image.open('./data/hw3_3.jpeg')
img.load()
data = np.asarray(img, dtype='float')/255
height, width, depth = data.shape
data = np.reshape(data, (-1, depth))  # pixels*RGB  (width*height) = pixels

K_list = [3, 5, 7, 10]

for k in K_list:
    k_means = K_means(K=k)
    k_means.initialize(data)
    k_means.minimize_J(data)

    # print_RGB_table(k_means, 'K_means')
    generate_img(k_means, 'K_means')

    gmm = GMM(k)
    gmm.initialize(k_means.rnk, k_means.mu, data)
    gmm.EM(data)
    gmm.plot_likelihood_log()

    # print_RGB_table(gmm, 'GMM')
    generate_img(gmm, 'GMM')
