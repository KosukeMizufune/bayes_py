import math

import numpy as np
from scipy import stats


class HierLogit:
    def __init__(self, n, k, j):
        self.beta = np.zeros((1, n, k))
        self.delta = np.zeros((1, j, k))
        self.sigma = np.identity(k).reshape(1, k, k)
        self.n = n
        self.k = k
        self.j = j
        self.accept = 0

    def estimate(self, y, x, z, index, it=10000, mu_delta=0, sigma_delta=0, df=6, scale=1, random=0.01):
        for _ in range(it):
            random_var = random * self.sigma[-1]
            beta_tmp = np.zeros((0, self.k))
            for i in range(self.n):
                y_ind = y[index == i]
                x_ind = x[index == i]
                z_ind = z[i]
                beta_tmp = np.append(beta_tmp, self.mh(
                    y_ind, x_ind, z_ind, self.beta[-1][i], self.delta[-1], self.sigma[-1], random_var), 0)
            self.beta = np.append(self.beta, beta_tmp.reshape(1, self.n, self.k), 0)
            delta = self.generate_delta(z, mu_delta, self.sigma[-1], sigma_delta).reshape(1, self.j, self.k)
            self.delta = np.append(self.delta, delta, 0)
            sig = self.generate_sigma(self.beta[-1], self.delta[-1], z, df, scale).reshape(1, self.k, self.k)
            self.sigma = np.append(self.sigma, sig, 0)

    def loglik(self, param, y, x):
        u = np.exp(np.dot(x, param))
        p = u / (1 + u)
        return np.dot(y, np.log(p)) + np.dot((1-y), np.log(1-p))

    def loglik_multinormal(self, y, z, delta, sigma):
        mu = np.dot(delta.T, z)
        return -0.5 * np.dot(np.dot((y - mu).T, np.linalg.inv(sigma)), y - mu)

    def mh(self, y, x, z, param, delta, sigma, random):
        beta_old = param
        beta_new = np.random.multivariate_normal(beta_old, random)
        loglik_new = self.loglik(beta_new, y, x)
        loglik_new_prior = self.loglik_multinormal(beta_new, z, delta, sigma)
        loglik_old = self.loglik(beta_old, y, x)
        loglik_old_prior = self.loglik_multinormal(beta_old, z, delta, sigma)
        accept_prob = np.exp(loglik_new + loglik_new_prior - loglik_old - loglik_old_prior)
        if math.isnan(accept_prob):
            accept_prob = -1
        u = np.random.uniform(0.0, 1.0, 1)
        if u < accept_prob:
            self.accept += 1
            return beta_new.reshape(1, self.k)
        else:
            return beta_old.reshape(1, self.k)

    def generate_delta(self, z, mu_delta, sigma, sigma_delta):
        z_pow = np.dot(z.T, z)
        tmp_var = np.linalg.inv(z_pow + sigma_delta)
        var = np.kron(sigma, tmp_var)
        delta_tilde = np.dot(np.dot(np.linalg.inv(z_pow), z.T), self.beta[-1])
        mean = np.dot(tmp_var, np.dot(z_pow, delta_tilde) + np.dot(sigma_delta, mu_delta))
        return np.random.multivariate_normal(mean.reshape(-1,), var)

    def generate_sigma(self, beta, delta, z, df, scale):
        eps = beta - np.dot(delta.T, z.T).T
        s = np.dot(eps.T, eps)
        return stats.invwishart.rvs(df + self.n, scale + s)

    def summary(self, keep, burnin):
        beta = self.beta[0::keep]
        sigma = self.sigma[0::keep]
        delta = self.delta[0::keep]
        print("delta")
        print(np.apply_along_axis(np.mean, 0, delta[burnin+1:]))
        print("sigma")
        print(np.apply_along_axis(np.mean, 0, sigma[burnin+1:]))
