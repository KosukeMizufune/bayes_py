import numpy as np
from scipy import stats


class GibbsSampler:
    def __init__(self):
        self.sigma = np.exp([0])  # 分散の逆数
        self.alpha = np.zeros(1)
        self.beta = np.zeros(1)

    def gibbs_simple_linear(self, y, x, it, alpha_0=1, beta_0=1, sigma_alpha_0=1, sigma_beta_0=1,
                            sigma_shape=1, sigma_scale=1):
        """

        :param y: 被説明変数
        :param x: 説明変数
        :param alpha_0: float, 切片項の平均の事前分布
        :param beta_0: float, 係数の平均の事前分布
        :param sigma_alpha_0: float, 切片項の分散の事前分布
        :param sigma_beta_0: float, 係数の分散の事前分布
        :param sigma_scale: float, 分散のscale parameter
        :param sigma_shape: float, 分散のshape parameter
        :param it: int, iterationの回数
        """
        for _ in range(it):
            self.alpha = np.append(self.alpha, self.generate_alpha(y, x, alpha_0, sigma_alpha_0))
            self.beta = np.append(self.beta, self.generate_beta(y, x, beta_0, sigma_beta_0))
            self.sigma = np.append(self.sigma, self.generate_sigma(y, x, sigma_shape, sigma_scale))

    def generate_alpha(self, y, x, mu, sigma):
        n = len(y)
        sigma_1 = 1 / (sigma + n * self.sigma[-1])
        alpha_1 = sigma_1 * (sigma * mu + self.sigma[-1] * np.sum(y - self.beta[-1] * x))
        return np.random.normal(alpha_1, np.sqrt(sigma_1))

    def generate_beta(self, y, x, mu, sigma):
        sigma_1 = 1 / (sigma + np.sum(x ** 2) * self.sigma[-1])
        beta_1 = sigma_1 * (sigma * mu + self.sigma[-1] * np.sum(x * y - self.alpha[-1] * x))
        return np.random.normal(beta_1, np.sqrt(sigma_1))

    def generate_sigma(self, y, x, alpha, beta):
        n = len(y)
        shape = alpha + n
        scale = beta + np.sum((y - self.alpha[-1] - self.beta[-1] * x) ** 2)
        return np.random.gamma(shape / 2, 2 / scale)


class GibbsMultiple:
    def __init__(self, dim_beta):
        self.beta = np.zeros((dim_beta, 1))
        self.sigma = np.exp([0])

    def gibbs_linear(self, y, x, it, beta_0, sigma_beta, alpha, theta):
        if len(x.shape) == 1:
            x = np.array(x).reshape(x.shape[0], 1)
        x = np.append(x, np.ones(x.shape[0]).reshape(x.shape[0], 1), axis=1)
        for i in range(it):
            self.beta = np.append(self.beta, self.generate_beta(y, x, beta_0, sigma_beta).reshape(x.shape[1], 1),
                                  axis=1)
            self.sigma = np.append(self.sigma, self.generate_sigma(y, x, alpha, theta))

    def generate_beta(self, y, x, mu, sigma):
        sigma_1 = np.linalg.inv(sigma + np.dot(x.T, x) * self.sigma[-1])
        beta_1 = np.dot(sigma_1, np.dot(sigma, mu) + np.dot(x.T, y) * self.sigma[-1])
        return np.random.multivariate_normal(beta_1, sigma_1)

    def generate_sigma(self, y, x, alpha, theta):
        n = len(y)
        shape = alpha + n
        scale = theta + np.dot((y - np.dot(x, self.beta[:, -1])).T, (y - np.dot(x, self.beta[:, -1])))
        return np.random.gamma(shape / 2, 2 / scale)


class GibbsProbit:
    def __init__(self, dim_beta, dim_choice, n):
        self.beta = np.zeros((dim_beta + dim_choice - 1, 1))
        self.z = [np.zeros((n, dim_choice - 1))]
        self.dim_beta = dim_beta
        self.dim_choice = dim_choice
        self.dim = dim_beta + dim_choice - 1
        self.sigma = [np.identity(self.dim_choice - 1)]

    def gibbs_linear(self, y, x, it, beta_0, sigma_beta, df, scale):
        x = self.transform_x(x)
        index = []
        ind_y = []
        ind_not = []
        for i in range(self.dim_choice - 1):
            ind = np.ones(self.dim_choice - 1, dtype=bool)
            ind[i] = False
            index.append(ind)
            ind_y.append(y[:, i] == 1)
            ind_not.append(y[:, i] == 0)
        for i in range(it):
            self.z.append(self.generate_z(y, x, self.beta[:, -1], index, ind_y, ind_not))
            self.beta = np.append(
                self.beta, self.generate_beta(y, x, self.z[-1], beta_0, sigma_beta).reshape(self.dim, 1), axis=1)
            self.sigma.append(self.generate_sigma(self.z[-1], x, self.beta[:, -1], df, scale))

    def generate_z(self, y, x, beta, ind, ind_y, ind_not):
        # 問題あり
        z = np.zeros((y.shape[0], self.dim_choice - 1))
        tmp = np.zeros(y.shape[0])
        utility = np.dot(x, beta)
        for i in range(self.dim_choice - 1):
            f = np.dot(np.linalg.inv(self.sigma[-1])[:, ind[i]][ind[i]], self.sigma[-1][:, i][ind[i]])
            m = utility[:, i] + np.dot(f.T, (self.z[-1][:, ind[i]] - utility[:, ind[i]]).T)
            tau = np.sqrt(self.sigma[-1][i][i] - np.dot(self.sigma[-1][i][ind[i]], f))
            can = np.append(self.z[-1][:, ind[i]], np.zeros((y.shape[0], 1)), axis=1)
            w_max = np.amax(can, axis=1)
            tmp[ind_y[i]] = self.tran_normal(m[ind_y[i]], tau, w_max[ind_y[i]], np.inf)
            tmp[ind_not[i]] = self.tran_normal(m[ind_not[i]], tau, -np.inf, w_max[ind_not[i]])
            z[:, i] = tmp
        return z

    def generate_beta(self, y, x, z, mu_beta, sigma_beta):
        c = np.linalg.cholesky(np.linalg.inv(self.sigma[-1]))
        z = z.reshape((y.shape[0], self.dim_choice - 1, 1))
        left_new = np.dot(c, z).reshape((y.shape[0] * (self.dim_choice - 1), 1))
        right_new = np.dot(c, x).reshape((y.shape[0] * (self.dim_choice - 1), self.dim))
        sig = np.linalg.inv(np.dot(right_new.T, right_new) + sigma_beta)
        mu = np.dot(sig, (np.dot(right_new.T, left_new).reshape(self.dim) + np.dot(sigma_beta, mu_beta)))
        return np.random.multivariate_normal(mu, sig)

    def generate_sigma(self, z, x, beta, df, scale):
        new_df = x.shape[0] + df
        eps = z - np.dot(x, beta)
        new_scale = scale + np.dot(eps.T, eps)
        result = stats.invwishart.rvs(new_df, new_scale)
        return result

    def transform_x(self, x):
        tmp = np.ndarray((x.shape[0], 0))
        for i in range(self.dim_choice - 1):
            tmp = np.append(tmp, x[:, 1 + i::self.dim_choice] - x[:, 0::self.dim_choice], axis=1)
            intercept = np.zeros((x.shape[0], self.dim_choice - 1))
            intercept[:, i] = 1
            tmp = np.append(tmp, intercept, axis=1)
        return tmp.reshape((x.shape[0], self.dim_choice - 1, self.dim))

    def tran_normal(self, mu, sigma, a, b):
        fa = stats.norm.cdf(x=a, loc=mu, scale=sigma)
        fb = stats.norm.cdf(x=b, loc=mu, scale=sigma)
        return stats.norm.ppf(fa + np.random.uniform(0, 1, len(mu)) * (fb - fa), mu, sigma)
