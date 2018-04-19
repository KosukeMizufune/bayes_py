import math

from numdifftools import Hessian
import numpy as np
from scipy import optimize


class MultiLogit:
    def __init__(self, dim_beta, dim_choice):
        self.beta = np.zeros((dim_beta + dim_choice - 1, 1))
        self.dim_beta = dim_beta
        self.dim_choice = dim_choice
        self.dim = dim_beta + dim_choice - 1

    def multi_logit(self, y, x, iter=10000, beta_0=0, sigma=1, scale=0.01):
        """
        多項ロジットのランダムウォークMHアルゴリズム推定。ランダムウォークの尺度パラメータにはGelman(2013)より2.4 / √dを採用。採択率は.1~.3が望ましい。
        :param y: np.array, 選択行列。全て[0,1]で列数が選択肢の数と一致する必要。
        :param x: np.array, 選択肢ごとの説明変数。列において選択肢ごとに項目がまとまっている必要がある。
        :param iter: MCMCサンプリング数
        :param beta_0:
        :param sigma:
        :param scale:
        :return: 
        """
        accept = 0
        for _ in range(iter):
            beta_new = np.random.multivariate_normal(self.beta[:, -1], scale)
            loglik_new = -self.loglik(beta_new, y, x)
            loglik_new_prior = self.loglik_multinormal(beta_new, beta_0, sigma)
            loglik_old = -self.loglik(self.beta[:, -1], y, x)
            loglik_old_prior = self.loglik_multinormal(self.beta[:, -1], beta_0, sigma)
            r = np.exp(loglik_new + loglik_new_prior - loglik_old - loglik_old_prior)
            accept_prob = min(1, r)
            if math.isnan(accept_prob):
                accept_prob = -1
            u = np.random.uniform(0.0, 1.0, 1)
            if u < accept_prob:
                self.beta = np.append(self.beta, beta_new.reshape(self.dim, 1), axis=1)
                accept += 1
            else:
                self.beta = np.append(self.beta, self.beta[:, -1].reshape(self.dim, 1), axis=1)
        return accept / iter

    def loglik(self, param, y, x):
        beta = param[self.dim_choice - 1:]
        intercept = np.append(param[:self.dim_choice - 1], [0])
        u = np.zeros((y.shape[0], self.dim_choice))
        choice_u = np.zeros(self.dim_choice)
        denominator = np.zeros(y.shape[0])
        col = np.arange(0, self.dim_beta * self.dim_choice) % self.dim_choice
        for i in range(self.dim_choice):
            u[:, i] = np.dot(x[:, col == i], beta) + intercept[i]
            choice_u[i] = np.dot(y[:, i], u[:, i])
            denominator += np.exp(u[:, i])
        likelihood = np.sum(choice_u) - np.sum(np.log(denominator))
        return -likelihood

    def loglik_multinormal(self, y, mu, sigma):
        """
        カーネルの部分だけ導出する
        :param y: 
        :param mu: 
        :param sigma: 
        :return: 
        """
        return -np.dot(np.dot((y - mu).T, sigma), y - mu) / 2

    def derive_hess(self, y, x):
        ml_param = optimize.minimize(self.loglik, np.zeros(5), args=(y, x), method="BFGS").x
        return np.linalg.inv(Hessian(self.loglik)(ml_param, y, x))
