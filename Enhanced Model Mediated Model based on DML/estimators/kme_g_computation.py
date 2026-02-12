import os
import sys
import numpy as np
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from estimators.base import Estimator
from nuisances.kme import _kme_conditional_mean_outcome
from utils.decorators import fitted


class KMEGComputation(Estimator):
    """Implementation of Kernel Mean Embedding G computation

    Args:
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, settings, verbose=0):
        super(KMEGComputation, self).__init__(settings=settings, verbose=verbose)

        self._crossfit = 0
        self.name = 'G_comp'

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
            m.reshape(n, 1)

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
        """Fits nuisance parameters to data"""
        t, m, x, y = self.resize(t, m, x, y)

        # 添加密度拟合
        self.fit_tx_density(t, x)
        self.fit_txm_density(t, x, m)

        self._fitted = True
        if self.verbose:
            print(f"Nuisance models fitted")

    def fit_tx_density(self, t, x):
        """拟合处置-协变量联合密度"""
        # 这里添加实际密度拟合逻辑
        # 为简单起见，我们使用高斯核密度估计
        from sklearn.neighbors import KernelDensity
        self._density_tx = KernelDensity(kernel='gaussian', bandwidth=0.5)
        tx = np.column_stack((t, x))
        self._density_tx.fit(tx)

    def fit_txm_density(self, t, x, m):
        """拟合处置-协变量-中介联合密度"""
        from sklearn.neighbors import KernelDensity
        self._density_txm = KernelDensity(kernel='gaussian', bandwidth=0.5)
        txm = np.column_stack((t, x, m))
        self._density_txm.fit(txm)

    @fitted
    def estimate(self, d, d_prime, t, m, x, y):
        """Estimates causal effect on data

        """
        t, m, x, y = self.resize(t, m, x, y)

        eta_d_d, eta_d_d_prime, eta_d_prime_d, eta_d_prime_d_prime = (
            _kme_conditional_mean_outcome(d,
                                          d_prime,
                                          y,
                                          t,
                                          m,
                                          x,
                                          self._settings))

        direct_effect = eta_d_prime_d - eta_d_d
        indirect_effect = eta_d_prime_d_prime - eta_d_prime_d
        total_effect = direct_effect + indirect_effect

        # 添加方差估计
        # 使用自举法计算置信区间
        n_boot = 100  # 自举样本数
        boot_results = {
            'mediated_response': np.zeros(n_boot),
            'direct_effect': np.zeros(n_boot),
            'indirect_effect': np.zeros(n_boot),
            'total_effect': np.zeros(n_boot)
        }

        n = len(y)
        for i in range(n_boot):
            # 创建自举样本
            idx = np.random.choice(n, n, replace=True)
            t_boot = t[idx]
            m_boot = m[idx]
            x_boot = x[idx]
            y_boot = y[idx]

            # 在自举样本上计算
            eta_boot = _kme_conditional_mean_outcome(d, d_prime, y_boot, t_boot, m_boot, x_boot, self._settings)
            eta_d_d_boot, eta_d_d_prime_boot, eta_d_prime_d_boot, eta_d_prime_d_prime_boot = eta_boot

            boot_results['mediated_response'][i] = eta_d_d_prime_boot
            boot_results['direct_effect'][i] = eta_d_prime_d_boot - eta_d_d_boot
            boot_results['indirect_effect'][i] = eta_d_prime_d_prime_boot - eta_d_prime_d_boot
            boot_results['total_effect'][i] = eta_d_prime_d_prime_boot - eta_d_d_boot

        # 计算标准差和置信区间
        mediated_response_se = np.std(boot_results['mediated_response'])
        margin_error = 1.96 * mediated_response_se  # 95%置信区间

        causal_effects = {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'mediated_response': eta_d_d_prime,
            'variance': mediated_response_se ** 2,
            'margin_error': margin_error
        }

        return causal_effects

    def _kme_conditional_mean_outcome(d, d_prime, y, t, m, x, settings):
        """计算条件平均结果"""

        # 计算核函数
        def gaussian_kernel(u, h=1.0):
            return np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2 * np.pi))

        # 计算处置核
        kernel_t_d = gaussian_kernel(t - d)
        kernel_t_d_prime = gaussian_kernel(t - d_prime)

        # 计算条件平均结果
        eta_d_d = np.sum(kernel_t_d * y) / np.sum(kernel_t_d)
        eta_d_d_prime = np.sum(kernel_t_d_prime * y) / np.sum(kernel_t_d_prime)
        eta_d_prime_d = np.sum(kernel_t_d_prime * y) / np.sum(kernel_t_d_prime)
        eta_d_prime_d_prime = np.sum(kernel_t_d_prime * y) / np.sum(kernel_t_d_prime)

        return eta_d_d, eta_d_d_prime, eta_d_prime_d, eta_d_prime_d_prime