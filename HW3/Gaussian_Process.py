import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


def exp_quad_kernel(x, y, params):
    # fixme: 自己重寫
    return params[0]*np.exp(-0.5*params[1]*np.subtract.outer(x, y)**2) + params[2] + params[3]*np.multiply.outer(x, y)


def RMSE(x, y):
    return np.sqrt(np.sum((x-y)**2)/len(x))


# load data
data = loadmat('./data/gp.mat')
X = data['x'].squeeze()
T = data['t'].squeeze()
train_x = X[:60]
test_x = X[60:]
train_t = T[:60]
test_t = T[60:]

x = np.linspace(0, 2, 300)
y = np.empty(300)
y1 = np.empty(300)
y2 = np.empty(300)

# set param
parameters = [[1, 4, 0, 0],
              [0, 0, 0, 1],
              [1, 4, 0, 5],
              [1, 32, 5, 5]]
beta_inv = 1


# use df to print
def print_table(t1, t2, t3):
    return pd.DataFrame(columns=[t1, t2, t3])


table = print_table("param", "training error", "testing error")


# draw the predictive distribution plot
def plot_gp(y1, y2, title, save=1):
    plt.plot(x, y, 'r-')
    plt.fill_between(x, y1, y2, facecolor='pink', edgecolor='none')
    plt.scatter(train_x, train_t, facecolors='none', edgecolors='b')  # plot the data point
    plt.title(str(title))
    plt.xlim(0, 2)
    plt.ylim(-10, 15)
    plt.xlabel('x')
    plt.ylabel('t', rotation=0)
    if save:
        plt.savefig('./images/gp_param' + str(p) + '.png')
    plt.show()


def GP(table, pars, C_inv, p=-1, iter=300):
    for i in range(iter):
        k = exp_quad_kernel(train_x, x[i], pars[p])
        c = exp_quad_kernel(x[i], x[i], pars[p]) + beta_inv
        y[i] = np.linalg.multi_dot([k, C_inv, train_t])
        std = np.sqrt(c - np.linalg.multi_dot([k.T, C_inv, k]))
        y1[i] = y[i] + std
        y2[i] = y[i] - std
    plot_gp(y1, y2, pars[p])
    # calculate the rms on training and test data
    train_y = np.empty(60)
    for i in range(60):
        k = exp_quad_kernel(train_x, train_x[i], pars[p])
        train_y[i] = np.linalg.multi_dot([k, C_inv, train_t])

    predict = np.empty(40)
    for i in range(40):
        k = exp_quad_kernel(train_x, test_x[i], pars[p])
        predict[i] = np.linalg.multi_dot([k, C_inv, train_t])
    return table.append(pd.Series(["("+str(pars[p])[1:-1]+")", RMSE(train_y, train_t), RMSE(predict, test_t)], index=["param", "training error", "testing error"]), ignore_index=True)


# iterate throw 4 set of paramaters
for p in range(4):
    C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters[p]) + beta_inv * np.identity(60))
    table = GP(table, parameters, C_inv, p)
print(table.to_string(index=False))


# tune hyperparameters with ARD
def dev_log_like(C_inv, C_dev, t):
    return -0.5 * np.trace(C_inv.dot(C_dev)) + 0.5 * np.linalg.multi_dot([t.T, C_inv, C_dev, C_inv, t])


parameters_ard = [[3, 6, 4, 5]]
dev_func = [0, 0, 0, 0]
learning_rate = 0.001


def ARD(params):
    while True:
        C_inv = np.linalg.inv(exp_quad_kernel(train_x, train_x, params[-1]) + beta_inv * np.identity(60))

        # update parameter
        dev_func[0] = dev_log_like(C_inv, np.exp(-0.5 * params[-1][1] * np.subtract.outer(train_x, train_x)**2), train_t)

        dev_func[1] = dev_log_like(C_inv, params[-1][0] * -0.5 * np.subtract.outer(train_x, train_x) * np.exp(-0.5 * params[-1][1] * np.subtract.outer(train_x, train_x)**2), train_t)

        dev_func[2] = dev_log_like(C_inv, np.full([60, 60], 1), train_t)
        dev_func[3] = dev_log_like(C_inv, np.multiply.outer(train_x, train_x), train_t)
        params.append([p + learning_rate * dev for p, dev in zip(params[-1], dev_func)])

        if np.max(np.abs(dev_func)) < 6:
            return params


def plot_param(par):
    plt.plot(par[:, 0], label='hyperparameter 0')
    plt.plot(par[:, 1], label='hyperparameter 1')
    plt.plot(par[:, 2], label='hyperparameter 2')
    plt.plot(par[:, 3], label='hyperparameter 3')
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("value", rotation=90)
    plt.show()


parameters_ard = ARD(parameters_ard)
params_ard = np.array(parameters_ard)
plot_param(params_ard)
table_ard = print_table("param", "training error", "testing error")
C_inv_ard = np.linalg.inv(exp_quad_kernel(train_x, train_x, parameters_ard[-1]) + beta_inv * np.identity(60))
table_ard = GP(table_ard, parameters_ard, C_inv_ard)
print(table_ard.to_string(index=False))

