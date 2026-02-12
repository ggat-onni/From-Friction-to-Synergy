import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy.stats import norm
from estimators.base import Estimator
import matplotlib.pyplot as plt
from nuisances.kme import _kme_cross_conditional_mean_outcomes
from utils.decorators import fitted

# 添加基础路径
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)


# 核函数保持不变
def gaussian_kernel(u):
    return np.exp(-0.5 * u ** 2) / (np.sqrt(2 * np.pi))


def gaussian_kernel_h(u, h_2):
    return (1 / (np.sqrt(h_2) * np.sqrt(2 * np.pi))) * np.exp((-0.5) / h_2 * (1.0 * u) ** 2)


def epanechnikov_kernel(u):
    condition = np.abs(u) <= 1
    return np.where(condition, 0.75 * (1 - u ** 2), 0)


class NeuralNetworkConditionalMean:
    """神经网络条件均值估计器"""

    def __init__(self, input_dim, hidden_layers=[64, 32], activation='relu',
                 learning_rate=0.1, epochs=100, batch_size=64, early_stop=True):
        self.model = self._build_model(input_dim, hidden_layers, activation, learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.input_dim = input_dim
        self.training_history = None  # 存储训练历史

    def _build_model(self, input_dim, hidden_layers, activation, learning_rate):
        """构建神经网络模型"""
        model = tf.keras.Sequential()

        # 添加输入层和隐藏层
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            model.add(tf.keras.layers.BatchNormalization())

        # 添加输出层
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )
        return model

    def fit(self, X, y):
        """训练神经网络"""
        callbacks = []
        if self.early_stop:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ))

        # 添加验证集
        val_split = 0.1 if self.early_stop else 0.0

        # 记录训练历史
        self.training_history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=0
        )
        return self.training_history

    def get_training_summary(self):
        """获取训练摘要"""
        if self.training_history is None:
            return "模型尚未训练"

        history = self.training_history.history
        epochs = len(history['loss'])

        # 获取最佳损失值
        best_train_loss = min(history['loss'])
        if 'val_loss' in history:
            best_val_loss = min(history['val_loss'])
        else:
            best_val_loss = None

        # 获取最终损失值
        final_train_loss = history['loss'][-1]

        summary = f"训练轮数: {epochs}/{self.epochs}\n"
        summary += f"最终训练损失: {final_train_loss:.6f}\n"
        summary += f"最佳训练损失: {best_train_loss:.6f}\n"

        if best_val_loss is not None:
            summary += f"最佳验证损失: {best_val_loss:.6f}\n"
            # 检查过拟合
            overfit_ratio = final_train_loss / best_val_loss
            if overfit_ratio < 0.8:
                summary += f"⚠️ 过拟合警告: 训练损失仅为验证损失的{overfit_ratio:.1%}\n"

        return summary

    def plot_training_history(self):
        """绘制训练历史图表"""
        if self.training_history is None:
            print("无训练历史可绘制")
            return

        history = self.training_history.history
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='训练损失')

        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='验证损失')

        plt.title('模型训练历史')
        plt.xlabel('训练轮数')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        plt.show()


    def predict(self, X):
        """进行预测"""
        return self.model.predict(X, verbose=0).flatten()


class NNDoubleMachineLearning(Estimator):
    """基于神经网络的双重机器学习实现

    使用神经网络替换KME DML中的条件回归部分
    """

    def __init__(self, settings, verbose=0):
        super(NNDoubleMachineLearning, self).__init__(settings=settings, verbose=verbose)

        self._crossfit = 0
        self._normalized = self._settings['normalized']
        self._sample_splits = self._settings['sample_splits']
        self.kernel = gaussian_kernel
        self.name = 'NN-DML'
        self._bandwidth_mode = self._settings['bandwidth_mode']
        self._epsilon = settings['epsilon']

        # 神经网络参数
        self.nn_settings = {
            'hidden_layers': settings.get('hidden_layers', [64, 32]),
            'activation': settings.get('activation', 'relu'),
            'learning_rate': settings.get('learning_rate', 0.1),
            'epochs': settings.get('epochs', 100),
            'batch_size': settings.get('batch_size', 64),
            'early_stop': settings.get('early_stop', True)
        }

    def evaluate_nn_models(self):
        """评估神经网络模型性能"""
        if not hasattr(self, 'nn_txm') or not hasattr(self, 'nn_tx'):
            print("神经网络模型未初始化")
            return

        print("\n" + "=" * 50)
        print("神经网络模型性能评估")
        print("=" * 50)

        print("\n模型: Y ~ T,X,M")
        print(self.nn_txm.get_training_summary())
        self.nn_txm.plot_training_history()

        print("\n模型: Y ~ T,X")
        print(self.nn_tx.get_training_summary())
        self.nn_tx.plot_training_history()





    def resize(self, t, m, x, y):
        """调整数据形状"""
        if len(y) != len(y.ravel()):
            raise ValueError("Multidimensional y is not supported")
        if len(t) != len(t.ravel()):
            raise ValueError("Multidimensional t is not supported")

        n = len(y)
        if len(x.shape) == 1:
            x = x.reshape(n, -1)
        if len(m.shape) == 1:
            m = m.reshape(n, -1)

        if n != len(x) or n != len(m) or n != len(t):
            raise ValueError("Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()

        return t, m, x, y

    def fit(self, t, m, x, y):
        """拟合基函数"""
        t, m, x, y = self.resize(t, m, x, y)

        self.fit_tx_density(t, x)
        self.fit_txm_density(t, x, m)
        self.fit_bandwidth(t)

        self._fitted = True

        if self.verbose:
            print(f"Nuisance models fitted")

    def _nn_conditional_mean_outcomes(self, d_vec, d_prime_vec, y, t, m, x):
        """使用神经网络估计条件期望结果

        替换原来的核方法条件均值估计
        """
        n = len(y)

        # 准备输入数据
        X_txm = np.column_stack((t, x, m))
        X_tx = np.column_stack((t, x))

        self.X_txm = X_txm
        self.X_tx = X_tx
        self.y_true = y

        # 创建神经网络模型
        nn_txm = NeuralNetworkConditionalMean(
            input_dim=X_txm.shape[1],
            **self.nn_settings
        )
        nn_tx = NeuralNetworkConditionalMean(
            input_dim=X_tx.shape[1],
            **self.nn_settings
        )

        # 训练神经网络模型
        nn_txm.fit(X_txm, y)
        nn_tx.fit(X_tx, y)



        # 为d值准备输入
        X_d_xm = np.column_stack((d_vec, x, m))
        X_d_prime_xm = np.column_stack((d_prime_vec, x, m))
        X_d_x = np.column_stack((d_vec, x))
        X_d_prime_x = np.column_stack((d_prime_vec, x))

        # 进行预测
        mu_d = nn_txm.predict(X_d_xm)
        mu_d_prime = nn_txm.predict(X_d_prime_xm)
        psi_d_d = nn_tx.predict(X_d_x)
        psi_d_d_prime = nn_tx.predict(X_d_prime_x)

        # 这些值在原始实现中相同
        psi_d_prime_d = psi_d_d.copy()
        psi_d_prime_d_prime = psi_d_d_prime.copy()

        return mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime


    def estimate(self, d, d_prime, t, m, x, y):
        """估计因果效应"""
        if self._bandwidth_mode == 'amse':
            self.fit_bandwidth(t)
            self.fit_amse_bandwidth(d, d_prime, t, m, x, y)
            return self._estimate(d, d_prime, t, m, x, y)
        else:
            return self._estimate(d, d_prime, t, m, x, y)

    def _estimate(self, d, d_prime, t, m, x, y):
        """使用神经网络进行估计"""
        n = t.shape[0]
        indirect_effects_fold = []

        # 创建占位符
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

            # 密度计算
            f_d_x = self._density_tx.pdf(x, d_vec) + 1e-10
            f_d_prime_x = self._density_tx.pdf(x, d_prime_vec) + 1e-10

            xm = np.hstack((x, m))
            f_d_xm = self._density_txm.pdf(xm, d_vec) + 1e-10
            f_d_prime_xm = self._density_txm.pdf(xm, d_prime_vec) + 1e-10

            # 使用神经网络估计条件均值
            mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                self._nn_conditional_mean_outcomes(d_vec, d_prime_vec, y, t, m, x)
            )
            # 评估神经网络模型
            if self.verbose > 0:
                self.evaluate_nn_models()

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
            # 初始化KFold进行样本分割
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

                # 训练密度模型
                self.fit_tx_density(t[train_idx], x[train_idx])
                self.fit_txm_density(t[train_idx], x[train_idx], m[train_idx])

                # 密度计算
                f_d_x = self._density_tx.pdf(x[test_idx], d_vec) + 1e-10
                f_d_prime_x = self._density_tx.pdf(x[test_idx], d_prime_vec) + 1e-10

                xm_test = np.hstack((x[test_idx], m[test_idx]))
                f_d_xm = self._density_txm.pdf(xm_test, d_vec) + 1e-10
                f_d_prime_xm = self._density_txm.pdf(xm_test, d_prime_vec) + 1e-10

                # 使用神经网络估计条件均值
                mu_d, mu_d_prime, psi_d_d, psi_d_d_prime, psi_d_prime_d, psi_d_prime_d_prime = (
                    self._nn_conditional_mean_outcomes(
                        d_vec, d_prime_vec,
                        y[test_idx],
                        t[test_idx],
                        m[test_idx],
                        x[test_idx]
                    )
                )

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

        # 计算平均分数
        my_dm_d = np.mean(y_d_m_d)
        my_d_prime_m_d_prime = np.mean(y_d_prime_m_d_prime)
        my_d_prime_m_d = np.mean(y_d_prime_m_d)
        my_d_m_d_prime = np.mean(y_d_m_d_prime)

        v_d_m_d_prime = self._bandwidth * np.mean((y_d_m_d_prime - my_d_m_d_prime) ** 2)

        confidence_level = 0.95
        # 计算临界值
        z_alpha_2 = norm.ppf(1 - (1 - confidence_level) / 2)

        # 计算误差范围
        margin_of_error = z_alpha_2 * np.sqrt(v_d_m_d_prime / (self._bandwidth * n))

        # 计算效应
        total = my_d_prime_m_d_prime - my_dm_d
        direct = my_d_prime_m_d - my_dm_d
        indirect = my_d_prime_m_d_prime - my_d_prime_m_d
        mediated_response = my_d_m_d_prime
        effects = np.array([total, direct, indirect, mediated_response])
        scaled_effects = effects * 4
        total_e, direct_e, indirect_e, mediated_r = scaled_effects

        causal_effects = {
            'total_effect': total,
            'direct_effect': direct,
            'indirect_effect': indirect,
            'total_effects': total_e,
            'direct_effects': direct_e,
            'indirect_effects': indirect_e,
            'mediated_response': mediated_r,
            'variance': v_d_m_d_prime,
            'margin_error': margin_of_error,
        }
        return causal_effects

    def fit_amse_bandwidth(self, d, d_prime, t, m, x, y):
        """拟合AMSE带宽"""
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