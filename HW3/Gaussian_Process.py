import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def exp_quad_kernel(x, y, params):
    return params[0]*np.exp(-0.5*params[1]*np.subtract.outer(x, y)**2) + params[2] + params[3]*np.multiply.outer(x, y)


def RMSE(x, y):
    return np.sqrt(np.sum((x-y)**2)/len(x))


data = loadmat('./data/gp.mat')
X = data['x'].squeeze()
T = data['t'].squeeze()
beta_inv = 1

train_x = X[:60]
test_x = X[60:]
train_t = T[:60]
test_t = T[60:]

x = np.linspace(0, 2, 300)
y = np.empty(300)
y1 = np.empty(300)
y2 = np.empty(300)

parameters = [[1, 4, 0, 0],
              [0, 0, 0, 1],
              [1, 4, 0, 5],
              [1, 32, 5, 5]]

f, axarr = plt.subplots(2, 2)
# table = PrettyTable(["parameters", "train error", "test error"])


def plot_gp(y1, y2, p):
    plt.plot(x, y, 'r-')
    plt.fill_between(x, y1, y2, facecolor='pink', edgecolor='none')
    plt.scatter(train_x, train_t, facecolors='none', edgecolors='b')  # plot the data point
    plt.title(str(parameters[p]))
    plt.xlim(0, 2)
    plt.ylim(-10, 15)
    plt.xlabel('x')
    plt.ylabel('t', rotation=0)
    plt.savefig('./images/gp_param' + str(p) + '.png')
    plt.show()


for p in range(4):
    C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters[p]) + beta_inv * np.identity(60))

    # plot the distribution
    for i in range(300):
        k = exp_quad_kernel(train_x, x[i], parameters[p])
        c = exp_quad_kernel(x[i], x[i], parameters[p]) + beta_inv
        y[i] = np.linalg.multi_dot([k, C_inv, train_t])
        std = np.sqrt(c - np.linalg.multi_dot([k.T, C_inv, k]))
        y1[i] = y[i] + std
        y2[i] = y[i] - std

    plot_gp(y1, y2, p)

    train_y = np.empty(60)
    for i in range(60):
        k = exp_quad_kernel(train_x, train_x[i], parameters[p])
        c = exp_quad_kernel(train_x[i], train_x[i], parameters[p])
        train_y[i] = np.linalg.multi_dot([k, C_inv, train_t])

    predict = np.empty(40)
    for i in range(40):
        k = exp_quad_kernel(train_x, test_x[i], parameters[p])
        c = exp_quad_kernel(test_x[i], test_x[i], parameters[p])
        predict[i] = np.linalg.multi_dot([k, C_inv, train_t])

    # table.add_row(["{"+str(parameters[p])[1:-1]+"}", RMSE(train_y, train_t), RMSE(predict, test_t)])

# print table

