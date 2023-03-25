import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron import Perceptron
from adaline import AdalineGD, AdalineSGD
from utils import plot_decision_regions


def get_iris_data(normalized=False):
    df = pd.read_csv('iris.data')
    y = df.iloc[0:99, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    X = df.iloc[0:99, [0, 2]].values

    if normalized:
        X_std = np.copy(X)
        X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) / X_std[:,0].std()
        X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) / X_std[:,1].std()
        return X_std, y

    return X, y


def perceptron_test():
    X, y = get_iris_data()

    # perceptron_data_scatter(X)
    # perceptron_error_count(X, y)
    perceptron_train_and_show(X, y)


def adaline_gd_test(normalized=False):
    X, y = get_iris_data(normalized)

    n_iter = 10

    eta1 = 0.01
    ada_perc1 = AdalineGD(eta=eta1, n_iter=n_iter)
    ada_perc1.fit(X, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    y_axis = ada_perc1.cost_ if normalized else np.log10(ada_perc1.cost_)
    ax[0].plot(range(1, len(ada_perc1.cost_) + 1),
               y_axis,
               marker='o')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('cost' if normalized else 'log10(cost)')
    ax[0].set_title(f'Adaline - eta={eta1}')

    eta2 = 0.0001
    ada_perc2 = AdalineGD(eta=eta2, n_iter=n_iter)
    ada_perc2.fit(X, y)

    ax[1].plot(range(1, len(ada_perc2.cost_) + 1),
               ada_perc2.cost_,
               marker='o')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('cost')
    ax[1].set_title(f'Adaline - eta={eta2}')

    plt.show()


def adaline_sgd_test():
    X, y = get_iris_data(normalized=True)
    ada_perc = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
    ada_perc.fit(X, y)

    plot_decision_regions(X, y, ada_perc)

    plt.title('Adaline SGD')
    plt.xlabel('x0')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada_perc.cost_) + 1), ada_perc.cost_, marker='o')
    plt.xlabel('epochs')
    plt.ylabel('avg error')
    plt.tight_layout()
    plt.show()


def perceptron_data_scatter(X):
    plt.scatter(X[0:49, 0], X[0:49, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[49:99, 0], X[49:99, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('x0')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.show()


def perceptron_error_count(X, y):
    perceptron = Perceptron(eta=0.1, n_iter=10)
    perceptron.fit(X, y)
    plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_,
             marker='o')
    plt.xlabel('epochs')
    plt.ylabel('update count')
    plt.show()


def perceptron_train_and_show(X, y):
    perceptron = Perceptron(eta=0.1, n_iter=10)
    perceptron.fit(X, y)
    plot_decision_regions(X, y, perceptron)
    plt.xlabel('x0')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # perceptron_test()
    # adaline_gd_test(normalized=True)
    adaline_sgd_test()