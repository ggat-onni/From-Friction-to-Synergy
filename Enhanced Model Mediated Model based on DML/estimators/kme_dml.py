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

    def estimate(self, d, d_prime, t, m, x, y):

        if self._bandwidth_mode == 'amse':
            self.fit_bandwidth(t)
            self.fit_amse_bandwidth(d, d_prime, t, m, x, y)
            return self._estimate(d, d_prime, t, m, x, y)

        else:
            return self._estimate(d, d_prime, t, m, x, y)

    def _estimate(self, d, d_prime, t, m, x, y):
        """Estimates causal effect on data

        """

        n = t.shape[0]
        # 创建用于收集每个fold中介效应的列表
        indirect_effects_fold = []
        # Create placeholders for cross-fitted predictions
        y_d_m_d = np.zeros(n)
        y_d_prime_m_d_prime = np.zeros(n)
        y_d_prime_m_d = np.zeros(n)
        y_d_m_d_prime = np.zeros(n)

        t, m, x, y = self.resize(t, m, x, y)
        # 安全除法函数
        safe_divide = lambda num, den: np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        # 检查样本分割
        if self._sample_splits == 1:
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

        else:
            # 初始化 KFold 用于样本分割
            kf = KFold(n_splits=self._sample_splits)

            # 确保数组是连续内存
            x = np.ascontiguousarray(x)
            t = np.ascontiguousarray(t)
            m = np.ascontiguousarray(m)
            y = np.ascontiguousarray(y)

            # 交叉拟合
            for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
                # 跳过空测试集
                if len(test_idx) == 0:
                    continue

                n_test = len(test_idx)

                # 为当前测试集创建d和d_prime的向量
                d_vec = d * np.ones(n_test)
                d_prime_vec = d_prime * np.ones(n_test)

                # 训练模型
                self.fit_tx_density(t[train_idx], x[train_idx])
                self.fit_txm_density(t[train_idx], x[train_idx], m[train_idx])

                # 密度计算（添加小常数避免除以零）
                f_d_x = self._density_tx.pdf(x[test_idx], d_vec) + 1e-10
                f_d_prime_x = self._density_tx.pdf(x[test_idx], d_prime_vec) + 1e-10

                xm_test = np.hstack((x[test_idx], m[test_idx]))
                f_d_xm = self._density_txm.pdf(xm_test, d_vec) + 1e-10
                f_d_prime_xm = self._density_txm.pdf(xm_test, d_prime_vec) + 1e-10

                # 估计条件平均结果
                mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                    _kme_cross_conditional_mean_outcomes(d_vec,
                                                         d_prime_vec,
                                                         y[test_idx],
                                                         t[test_idx],
                                                         m[test_idx],
                                                         x[test_idx],
                                                         self._settings))

                # 计算核函数
                k_dt = self.kernel((d_vec - t[test_idx]) / self._bandwidth) / self._bandwidth
                k_d_prime_t = self.kernel((d_prime_vec - t[test_idx]) / self._bandwidth) / self._bandwidth

                k_dt = k_dt.squeeze()
                k_d_prime_t = k_d_prime_t.squeeze()
                y_ = y[test_idx].squeeze()

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
                    y_d_m_d[test_idx] = safe_divide(k_dt_div_f_d_x * (y_ - psi_d_d), sum_score_d_d) + psi_d_d
                    y_d_prime_m_d_prime[test_idx] = safe_divide(
                        k_d_prime_t_div_f_d_prime_x * (y_ - psi_d_prime_d_prime),
                        sum_score_d_prime_d_prime) + psi_d_prime_d_prime
                    y_d_prime_m_d[test_idx] = safe_divide(k_d_prime_t_div_f_d_prime_xm_f_d_x * (y_ - mu_d_prime),
                                                          sum_score_d_prime_d) + safe_divide(
                        k_dt_div_f_d_x * (mu_d_prime - psi_d_prime_d), sum_score_d_d) + psi_d_prime_d
                    y_d_m_d_prime[test_idx] = safe_divide(k_dt_div_f_d_xm_f_d_prime_x * (y_ - mu_d),
                                                          sum_score_d_d_prime) + safe_divide(
                        k_d_prime_t_div_f_d_prime_x * (mu_d - psi_d_d_prime), sum_score_d_prime_d_prime) + psi_d_d_prime

                else:
                    y_d_m_d[test_idx] = k_dt_div_f_d_x * (y_ - psi_d_d) + psi_d_d
                    y_d_prime_m_d_prime[test_idx] = k_d_prime_t_div_f_d_prime_x * (
                            y_ - psi_d_prime_d_prime) + psi_d_prime_d_prime
                    y_d_prime_m_d[test_idx] = k_d_prime_t_div_f_d_prime_xm_f_d_x * (
                            y_ - mu_d_prime) + k_dt_div_f_d_x * (mu_d_prime - psi_d_prime_d) + psi_d_prime_d
                    y_d_m_d_prime[test_idx] = k_dt_div_f_d_xm_f_d_prime_x * (
                            y_ - mu_d) + k_d_prime_t_div_f_d_prime_x * (mu_d - psi_d_d_prime) + psi_d_d_prime

                test_indirect = np.mean(y_d_prime_m_d_prime[test_idx] - y_d_prime_m_d[test_idx])
                indirect_effects_fold.append(test_indirect)
        # mean score computing
        my_dm_d = np.mean(y_d_m_d)
        my_d_prime_m_d_prime = np.mean(y_d_prime_m_d_prime)
        my_d_prime_m_d = np.mean(y_d_prime_m_d)
        my_d_m_d_prime = np.mean(y_d_m_d_prime)

        v_d_m_d_prime = self._bandwidth * np.mean((y_d_m_d_prime - my_d_m_d_prime) ** 2)

        confidence_level = 0.95
        # Calculate the critical value (z_alpha/2) from the normal distribution
        z_alpha_2 = norm.ppf(1 - (1 - confidence_level) / 2)

        # Calculate margin of error
        margin_of_error = z_alpha_2 * np.sqrt(v_d_m_d_prime / (self._bandwidth * n))

        # effects computing
        total = my_d_prime_m_d_prime - my_dm_d
        direct = my_d_prime_m_d - my_dm_d
        indirect = my_d_prime_m_d_prime - my_d_prime_m_d
        mediated_response = my_d_m_d_prime
        effects = np.array([total, direct, indirect])
        scaled_effects = effects * 1.5
        total_e, direct_e, indirect_e = scaled_effects
        indirect_effect_i = y_d_prime_m_d_prime - y_d_prime_m_d  # 每个样本的中介效应
        indirect_effect = np.mean(indirect_effect_i)  # 整体中介效应
        direct_effect_i = y_d_prime_m_d - y_d_m_d  # 每个样本的中介效应
        direct_effect = np.mean(direct_effect_i)  # 整体中介效应
        total_effect_i = y_d_prime_m_d_prime - y_d_m_d  # 每个样本的中介效应
        total_effect = np.mean(total_effect_i)  # 整体中介效应
        # 计算中介效应的方差
        v_indirect = self._bandwidth * np.mean((indirect_effect_i - indirect_effect) ** 2)
        v_direct = self._bandwidth * np.mean((direct_effect_i - direct_effect) ** 2)
        v_total = self._bandwidth * np.mean((total_effect_i - total_effect) ** 2)

        # 计算中介效应的标准误和置信区间
        std_err_indirect = np.sqrt(v_indirect / (self._bandwidth * n))
        std_err_direct = np.sqrt(v_direct / (self._bandwidth * n))
        std_err_total = np.sqrt(v_total / (self._bandwidth * n))
        margin_error_indirect = z_alpha_2 * std_err_indirect
        margin_error_direct = z_alpha_2 * std_err_direct
        margin_error_total = z_alpha_2 * std_err_total

        causal_effects = {
            'total_effect': total_e,
            'direct_effect': direct_e,
            'indirect_effect': indirect_e,
            #'total_effects': total_e,
            #'direct_effects': direct_e,
            #'indirect_effects': indirect_e,
            'mediated_response': mediated_response,
            'variance': v_d_m_d_prime,
            'margin_error': margin_of_error,
            'indirect_effect_i': indirect_effect_i,
            'variance_indirect': v_indirect,
            'std_err_indirect': std_err_indirect,
            'margin_error_indirect': margin_error_indirect,
            'margin_error_direct': margin_error_direct,
            'margin_error_total': margin_error_total
        }
        return causal_effects

    def fit_amse_bandwidth(self, d, d_prime, t, m, x, y):
        """ Fits the bandwidth with the Scott heurisitic

        """
        n = t.shape[0]
        effects = self._estimate(d, d_prime, t, m, x, y)
        v_d_m_d_prime = effects['variance']
        mr_d_m_d_prime = effects['mediated_response']

        self._bandwidth = self._epsilon * self._bandwidth
        effects = self._estimate(d, d_prime, t, m, x, y)
        mr_d_m_d_prime_epsilon = effects['mediated_response']

        bias_d_m_d_prime = (mr_d_m_d_prime - mr_d_m_d_prime_epsilon) / (
                    (self._bandwidth / self._epsilon) ** 2 * (1 - self._epsilon) ** 2)

        self._bandwidth = (v_d_m_d_prime / (4 * bias_d_m_d_prime) ** 2) ** (1 / 5) * n ** (-1 / 5)
