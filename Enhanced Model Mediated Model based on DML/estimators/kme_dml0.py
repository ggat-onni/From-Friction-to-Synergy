import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np

from estimators.base import Estimator
from nuisances.kme import _kme_cross_conditional_mean_outcomes
from utils.decorators import fitted
from sklearn.model_selection import KFold
from scipy.stats import norm


def gaussian_kernel(u):
    return np.exp(-0.5 * u ** 2) / (np.sqrt(2 * np.pi))


def gaussian_kernel_h(u, h_2):
    return (1 / (np.sqrt(h_2) * np.sqrt(2 * np.pi))) * np.exp((-0.5) / h_2 * (1.0 * u) ** 2)


def gaussian_k_bar(u):
    return (1 / (np.sqrt(4 * np.pi))) * np.exp(.25 * np.linalg.norm(1.0 * u) ** 2)


def epanechnikov_kernel(u):
    condition = np.abs(u) <= 1
    return np.where(condition, 0.75 * (1 - u ** 2), 0)


class KMEDoubleMachineLearning(Estimator):
    """Implementation of double machine learning

    Args:
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, settings, verbose=0):
        super(KMEDoubleMachineLearning, self).__init__(settings=settings, verbose=verbose)

        self._crossfit = 0
        self._normalized = self._settings['normalized']
        self._sample_splits = self._settings['sample_splits']
        self.kernel = gaussian_kernel
        self.name = 'DML'
        self._bandwidth_mode = self._settings['bandwidth_mode']
        self._epsilon = settings['epsilon']

    def resize(self, t, m, x, y):
        """Resize data for the right shape

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous
        """
        if len(y) != len(y.ravel()):
            raise ValueError("Multidimensional y is not supported")
        if len(t) != len(t.ravel()):
            raise ValueError("Multidimensional t is not supported")

        n = len(y)
        if len(x.shape) == 1:
            x.reshape(n, 1)
        if len(m.shape) == 1:
            m = m.reshape(n, 1)

        if n != len(x) or n != len(m) or n != len(t):
            raise ValueError(
                "Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()

        return t, m, x, y

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous

        """
        t, m, x, y = self.resize(t, m, x, y)

        self.fit_tx_density(t, x)
        self.fit_txm_density(t, x, m)
        self.fit_bandwidth(t)

        self._fitted = True

        if self.verbose:
            print(f"Nuisance models fitted")

    def estimate(self, d, d_prime, t, m, x, y, ids=None):
        """
        估计因果效应（使用全样本）

        Parameters
        ----------
        d, d_prime : float
            处置水平
        t, m, x, y : arrays
            数据向量
        ids : array-like, optional
            面板ID（用于聚类稳健标准误）
        """
        # 如果未提供ids，创建虚拟ID（每个观测值独立）
        if ids is None:
            ids = np.arange(len(y))

        # 确保ID向量与数据长度匹配
        if len(ids) != len(y):
            raise ValueError("IDs length must match data length")

        if self._bandwidth_mode == 'amse':
            self.fit_bandwidth(t)
            # 将ids传递给fit_amse_bandwidth
            self.fit_amse_bandwidth(d, d_prime, t, m, x, y, ids)
            return self._estimate(d, d_prime, t, m, x, y, ids)
        else:
            return self._estimate(d, d_prime, t, m, x, y, ids)

    def _estimate(self, d, d_prime, t, m, x, y, ids):
        """Estimates causal effect on data (using full sample)

        """
        n = t.shape[0]

        # 创建用于存储结果的数组
        y_d_m_d = np.zeros(n)
        y_d_prime_m_d_prime = np.zeros(n)
        y_d_prime_m_d = np.zeros(n)
        y_d_m_d_prime = np.zeros(n)

        t, m, x, y = self.resize(t, m, x, y)

        # 安全除法函数
        safe_divide = lambda num, den: np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        # 创建与t相同形状的d和d_prime向量
        d_vec = d * np.ones_like(t)
        d_prime_vec = d_prime * np.ones_like(t)

        # 密度计算（添加小常数避免除以零）
        f_d_x = self._density_tx.pdf(x, d_vec) + 1e-10
        f_d_prime_x = self._density_tx.pdf(x, d_prime_vec) + 1e-10

        xm = np.hstack((x, m))
        f_d_xm = self._density_txm.pdf(xm, d_vec) + 1e-10
        f_d_prime_xm = self._density_txm.pdf(xm, d_prime_vec) + 1e-10

        # 估计条件平均结果
        mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
            _kme_cross_conditional_mean_outcomes(d_vec,
                                                 d_prime_vec,
                                                 y,
                                                 t,
                                                 m,
                                                 x,
                                                 self._settings))

        # 计算核函数
        k_dt = self.kernel((d_vec - t) / self._bandwidth) / self._bandwidth
        k_d_prime_t = self.kernel((d_prime_vec - t) / self._bandwidth) / self._bandwidth

        k_dt = k_dt.squeeze()
        k_d_prime_t = k_d_prime_t.squeeze()
        y_ = y.squeeze()

        # 使用安全除法
        k_dt_div_f_d_x = safe_divide(k_dt, f_d_x)
        k_d_prime_t_div_f_d_prime_x = safe_divide(k_d_prime_t, f_d_prime_x)
        k_d_prime_t_div_f_d_prime_xm_f_d_x = safe_divide(k_d_prime_t * f_d_xm, (f_d_prime_xm * f_d_x))
        k_dt_div_f_d_xm_f_d_prime_x = safe_divide(k_dt * f_d_prime_xm, (f_d_xm * f_d_prime_x))

        # 计算分数
        if self._normalized:
            sum_score_d_d = np.mean(k_dt_div_f_d_x)
            sum_score_d_prime_d_prime = np.mean(k_d_prime_t_div_f_d_prime_x)
            sum_score_d_prime_d = np.mean(k_d_prime_t_div_f_d_prime_xm_f_d_x)
            sum_score_d_d_prime = np.mean(k_dt_div_f_d_xm_f_d_prime_x)

            # 使用安全除法
            y_d_m_d = safe_divide(k_dt_div_f_d_x * (y_ - psi_d_d), sum_score_d_d) + psi_d_d
            y_d_prime_m_d_prime = safe_divide(k_d_prime_t_div_f_d_prime_x * (y_ - psi_d_prime_d_prime),
                                              sum_score_d_prime_d_prime) + psi_d_prime_d_prime
            y_d_prime_m_d = safe_divide(k_d_prime_t_div_f_d_prime_xm_f_d_x * (y_ - mu_d_prime),
                                        sum_score_d_prime_d) + safe_divide(
                k_dt_div_f_d_x * (mu_d_prime - psi_d_prime_d), sum_score_d_d) + psi_d_prime_d
            y_d_m_d_prime = safe_divide(k_dt_div_f_d_xm_f_d_prime_x * (y_ - mu_d),
                                        sum_score_d_d_prime) + safe_divide(
                k_d_prime_t_div_f_d_prime_x * (mu_d - psi_d_d_prime), sum_score_d_prime_d_prime) + psi_d_d_prime

        else:
            y_d_m_d = k_dt_div_f_d_x * (y_ - psi_d_d) + psi_d_d
            y_d_prime_m_d_prime = k_d_prime_t_div_f_d_prime_x * (y_ - psi_d_prime_d_prime) + psi_d_prime_d_prime
            y_d_prime_m_d = k_d_prime_t_div_f_d_prime_xm_f_d_x * (y_ - mu_d_prime) + k_dt_div_f_d_x * (
                    mu_d_prime - psi_d_prime_d) + psi_d_prime_d
            y_d_m_d_prime = k_dt_div_f_d_xm_f_d_prime_x * (y_ - mu_d) + k_d_prime_t_div_f_d_prime_x * (
                    mu_d - psi_d_d_prime) + psi_d_d_prime

        # mean score computing
        my_dm_d = np.mean(y_d_m_d)
        my_d_prime_m_d_prime = np.mean(y_d_prime_m_d_prime)
        my_d_prime_m_d = np.mean(y_d_prime_m_d)
        my_d_m_d_prime = np.mean(y_d_m_d_prime)

        # 计算中介响应方差和置信区间
        v_d_m_d_prime = self._bandwidth * np.mean((y_d_m_d_prime - my_d_m_d_prime) ** 2)
        confidence_level = 0.95
        z_alpha_2 = norm.ppf(1 - (1 - confidence_level) / 2)
        margin_of_error = z_alpha_2 * np.sqrt(v_d_m_d_prime / (self._bandwidth * n))

        # 计算中介效应向量
        indirect_effect_vector = y_d_prime_m_d_prime - y_d_prime_m_d
        indirect_effect = np.mean(indirect_effect_vector)

        # 计算中介效应的聚类稳健标准误
        se_indirect = self.calculate_cluster_se(indirect_effect_vector, ids)
        me_indirect = z_alpha_2 * se_indirect
        ci_lower = indirect_effect - me_indirect
        ci_upper = indirect_effect + me_indirect

        # effects computing
        total = my_d_prime_m_d_prime - my_dm_d
        direct = my_d_prime_m_d - my_dm_d
        indirect = indirect_effect
        mediated_response = my_d_m_d_prime

        # 效应缩放（根据之前代码逻辑）
        effects = np.array([total, direct, indirect])
        scaled_effects = effects * 4
        total_e, direct_e, indirect_e = scaled_effects

        causal_effects = {
            'total_effect': total,
            'direct_effect': direct,
            'indirect_effect': indirect,
            'total_effects': total_e,
            'direct_effects': direct_e,
            'indirect_effects': indirect_e,
            'mediated_response': mediated_response,
            'variance': v_d_m_d_prime,
            'margin_error': margin_of_error,
            'variance_indirect': np.var(indirect_effect_vector),
            'se_indirect': se_indirect,
            'me_indirect': me_indirect,
            'ci_lower_indirect': ci_lower,
            'ci_upper_indirect': ci_upper
        }

        return causal_effects

    def calculate_cluster_se(self, effect_vector, ids):
        """
        计算聚类稳健标准误

        Parameters:
        effect_vector: 每个观测值的中介效应贡献
        ids: 面板ID向量
        """
        n = len(effect_vector)

        # 如果没有IDs或所有ID唯一，使用普通标准误
        if ids is None or len(np.unique(ids)) == n:
            return np.std(effect_vector) / np.sqrt(n)

        # 按ID分组计算组内均值
        unique_ids = np.unique(ids)
        cluster_means = []
        cluster_sizes = []

        for id_val in unique_ids:
            mask = (ids == id_val)
            cluster_vals = effect_vector[mask]
            cluster_means.append(np.mean(cluster_vals))
            cluster_sizes.append(len(cluster_vals))

        # 计算聚类稳健方差
        overall_mean = np.mean(effect_vector)
        cluster_effects = np.array(cluster_means) - overall_mean
        cluster_var = np.sum(np.square(cluster_effects) * np.array(cluster_sizes))

        # 小样本修正因子
        M = len(unique_ids)
        correction = (n - 1) / (n - M) * M / (M - 1)

        se = np.sqrt(cluster_var / n) * correction
        return se

    def fit_amse_bandwidth(self, d, d_prime, t, m, x, y, ids):
        """ Fits the bandwidth with the Scott heurisitic

        """
        n = t.shape[0]
        # 使用当前带宽估计一次
        effects = self._estimate(d, d_prime, t, m, x, y, ids)
        v_d_m_d_prime = effects['variance']
        mr_d_m_d_prime = effects['mediated_response']

        # 保存原始带宽
        original_bandwidth = self._bandwidth
        # 调整带宽
        self._bandwidth = self._epsilon * original_bandwidth
        # 用调整后的带宽再估计一次
        effects_epsilon = self._estimate(d, d_prime, t, m, x, y, ids)
        mr_d_m_d_prime_epsilon = effects_epsilon['mediated_response']

        # 计算偏差
        bias_d_m_d_prime = (mr_d_m_d_prime - mr_d_m_d_prime_epsilon) / (
                    (original_bandwidth / self._epsilon) ** 2 * (1 - self._epsilon) ** 2)

        # 计算AMSE最优带宽
        self._bandwidth = (v_d_m_d_prime / (4 * bias_d_m_d_prime) ** 2) ** (1 / 5) * n ** (-1 / 5)