# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:04:04 2020
Last update Sunday Oct 27 11:30 am 2023

该文件提供了主要的双偏差机器学习估计器，用于
连续处理的主要双分层机器学习估计器。 类 "DDMLCT "在以下情况下执行估计
.fit方法时执行估算。

DDMLCT 通过传递 2 个模型（如 sklearn 模型）进行初始化，这些模型有
.fit 和 .predict 方法。 一个用于估计广义倾向得分（GPS），一个用于估计伽马值。
模型 1 用于估计伽马值，模型 2 用于估计伽马值。
用于估计 GPS

DDMLCT 通过对每个 t 值进行一次交叉拟合来节省计算时间。
交叉拟合的每一褶都拟合一次，从而节省了计算时间。 由于 Colangelo 和 Lee（2022 年）中的 K 神经网络
中的 K 神经网络使用取决于 t 的唯一损失，因此我们必须定义另一个 "NN_DDMLCT "类。
除此以外，我们还进行了其他调整、两个类别几乎完全相同。

对所用软件包的注释：
    -copy用于复制用于启动 DDMLCT 的模型。
    -pandas 用于重新缩放，以重新缩放非哑铃模型。
    -scipy.stats.norm用于计算高斯核
    -numpy 用于存储大部分数据和属性。 如果数据
     如果数据以 pandas 数据帧的形式传递给 .fit 方法，则会转换为 numpy 数组
     数组。

"""
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import norm
import pandas as pd
import copy
import sklearn
from scipy.optimize import minimize
from scipy.stats import norm
import random
from sklearn.metrics import r2_score

# This function evaluates the gaussian kernel wtih bandwidth h at point x
def gaussian_kernel(x, h):
    k = (1 / h) * norm.pdf((x / h), 0, 1)
    return k


# This function evaluates the epanechnikov kernel
def e_kernel(x, h):
    k = (1 / h) * (3 / 4) * (1 - ((x / h) ** 2))
    k = k * (abs(x / h) <= 1)
    return k

def residual_analysis(self, Xf, XT, Xt, Y, I, I_C, L):
        """
        残差分析函数：计算预测值和残差。

        参数:
        - Xf: 额外的特征矩阵 (numpy array)
        - XT: 处理变量的矩阵 (numpy array)
        - Xt: 目标特征矩阵 (numpy array)
        - Y: 结果变量 (numpy array)
        - I: 当前折的索引 (list)
        - I_C: 补集索引（排除当前折）(list)
        - L: 当前模型的索引

        返回:
        - Y_pred: 模型对 Y 的预测值 (numpy array)
        - residuals: 残差，即真实值 Y 与预测值 Y_pred 的差 (numpy array)
        """
        # 训练阶段：若模型尚未训练，使用当前训练集 (I_C) 拟合
        if self.naive_count < self.L:
            X_train = np.column_stack((XT[I_C], Xf[I_C]))
            Y_train = Y[I_C]
            self.gamma_models.append(self.model1.fit(X_train, Y_train))
            self.naive_count += 1

        # 预测阶段：使用训练好的模型在验证集（当前折 I）上进行预测
        X_test = np.column_stack((Xt[I], Xf[I]))
        Y_pred = self.gamma_models[L].predict(X_test)

        # 计算残差：真实值减去预测值
        residuals = Y[I] - Y_pred

        return Y_pred, residuals

# The class which defines the proposed estimator.
class DDMLCT:
    """
    DDMLCT 通过传递 2 个模型（如 sklearn 模型）进行初始化，这些模型有
    .fit 和 .predict 方法。 一个用于估计广义倾向得分（GPS），一个用于估计伽马值。
    模型 1 用于估计伽马值，模型 2 用于估计伽马值。
    用于估计 GPS

    Parameters
    ----------

    model1: 估计伽马值的机器学习模型
    model2: 用于估算广义倾向得分的机器学习模型

    Attributes
    ----------
    beta： 一个数组，包含每个 t 的剂量-反应函数的所有估计值
    std_errors： 一个数组，包含与 beta 值相对应的所有标准误差

    Methods
    -------
    .fit: 是用户应该调用的唯一方法。 给定协变量 X
    结果 Y、治疗 T、评估估计值的 t 值列表、
    交叉拟合的子样本数 L、带宽选择 h、是否
    添加基函数以及是否标准化。
    """

    def __init__(self, model1, model2):
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.beta = np.array(())
        self.std_errors = np.array(())
        self.Vt = np.array(())
        self.Bt = np.array(())
        self.summary = None
        self.scaling = {'mean_Y': 0,
                        'sd_Y': 1,
                        'mean_T': 0,
                        'sd_T': 1}
        self.naive_count = 0
        self.L = 5
        self.gamma_models = []
        self.augment_indices = None  # 新增属性

    # 每次调用拟合方法时，都会调用重置，这样 .fit 就可以
    # 在同一对象上多次调用，并可能出现重叠。
    # t 的值。
    def augment(self, X, T, ind=None):
        T = T.reshape(len(T), 1)
        XT = np.column_stack((T, (T ** 2), (T ** 3), T * X))
        Xf = np.column_stack((X, X ** 2, X ** 3))
        Xf = np.unique(Xf, axis=1)
        if ind is None:
            XT, ind = np.unique(XT, axis=1, return_index=True)
        else:
            XT = XT[:, ind]
        self.augment_indices = ind  # 保存索引到实例属性
        return XT, Xf, ind

    def reset(self):
        self.beta = np.array(())
        self.std_errors = np.array(())
        self.Vt = np.array(())
        self.Bt = np.array(())
        self.summary = None
        self.scaling = {'mean_Y': 0,
                        'sd_Y': 1,
                        'mean_T': 0,
                        'sd_T': 1}
        self.naive_count = 0
        self.L = 5

    # 仅根据估计值定义 "Naive "机器学习估计器。
    # 定义 "Naive "机器学习估计器。 该函数只对每个子样本拟合一次，然后将拟合的 # 模型重复用于其他值 t，以增加计算量。
    # 对其他 t 值重复使用，以提高计算效率。
    # 效率。 naive_count" 变量会对此进行跟踪。 一旦它递增
    # 到 L 的水平，则表明模型已经在每个子样本中拟合过了。
    # 子样本。
    def naive(self, Xf, XT, Xt, Y, I, I_C, L):
        if self.naive_count < self.L:
            self.gamma_models.append(self.model1.fit(np.column_stack((XT[I_C], Xf[I_C])), Y[I_C]))
            self.naive_count += 1
        gamma = self.gamma_models[L].predict(np.column_stack((Xt[I], Xf[I])))
        # if sdml==False:
        #     if self.naive_count < self.L:
        #         self.gamma_models.append(self.model1.fit(np.column_stack((XT[I_C],Xf[I_C])),Y[I_C]))
        #         self.naive_count +=1
        #         gamma = self.gamma_models[L].predict(np.column_stack((Xt[I],Xf[I])))
        # else:
        #     gamma_model = self.model1.fit(np.column_stack((XT[I_C],Xf[I_C])),Y[I_C])
        #     gamma = gamma_model.predict(np.column_stack((Xt[I],Xf[I])))
        return gamma


    # g 是内核平滑函数，用于对 GPS 进行估计。
    # 被建模。
    def ipw(self, Xf, g, I, I_C):
        self.model2.fit(Xf[I_C], g[I_C])
        gps = self.model2.predict(Xf[I])

        return gps

    # 对于给定的 t，调用此函数来估计每个子样本
    # 交叉拟合的结果。 如果 L=5，则每个 t 都要调用该函数 5 次。
    def fit_L(self, Xf, XT, Xt, Y, g, K, I, I_C, L):
        gamma = self.naive(Xf, XT, Xt, Y, I, I_C, L)
        gps = self.ipw(Xf, g, I, I_C)
        self.kept = np.concatenate((self.kept, I))

        # Compute the summand
        psi = np.mean(gamma) + np.mean(((Y[I] - gamma) * (K[I] / gps)))

        # 对所有指数进行平均，得出beta的估计值
        beta_hat = np.mean(psi)


        return beta_hat, gamma, gps

    # 在给定的 t 列表中，每个 t 都会被调用此函数。
    # trep 是重复 n 次的 t 值。 XT 是
    # X、T 和添加的基函数的矩阵，这些基函数可能是 T 的函数。
    # Xf 是用于 GPS 估算的矩阵。
    # 用于 GPS 估算的矩阵。 如果使用附加基函数，则与 X 不同。
    # 它必须单独给出，否则
    # 否则就无法确定哪些变量取决于 T，哪些不取决于 T。
    def fit_t(self, Xf, T, Y, trep, L, XT, Xt, trep_sdml=None, sdml=False):

        self.kept = np.array((), dtype=int)  # 用于修剪，目前尚未实施

        T_t = T - trep

        g = gaussian_kernel(T_t, self.h)
        K = e_kernel(T_t, self.h)  #
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)

        mse_list = []  # 用于存储每折的 MSE
        r2_list = []  # 存储当前 t 值下所有子样本的 R^2

        # 对所有 L 个子样本进行迭代。 在拟合函数中定义了 I_split，以便对所有 t 的选择使用相同的分割。
        for i in range(L):
            if L == 1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I = self.I_split[i]
                # 将补集定义为所有其他集合的联合集
                I_C = [x for x in np.arange(self.n) if x not in I]

            beta_hat[i], gamma[I], gps[I] = self.fit_L(Xf, XT, Xt, Y, g, K, I, I_C, i)

            # 计算每折的 MSE 并存储
            mse = mean_squared_error(Y[I], gamma[I])
            mse_list.append(mse)
            r2 = r2_score(Y[I], gamma[I])
            r2_list.append(r2)
            beta_hat[i] = 16 * beta_hat[i]
            # 现在，我们对所有子样本进行平均，得出估计值和标准误差。
            # 计算平均 MSE
        avg_mse = np.mean(mse_list)
        avg_r2 = np.mean(r2_list)  # 当前 t 值的平均 R^2
        self.mse_list.append(avg_mse)  # 累积 MSE 到类属性中
        self.r2_list.append(avg_r2)

        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta, beta_hat)
        IF = (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept]) + gamma[self.kept] - beta_hat
        std_error = np.sqrt((1 / ((self.n) ** 2)) * np.sum(IF ** 2))
        self.Bt = np.append(self.Bt, (1 / (self.n * (self.h ** 2))) * (
            np.sum((K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept]))))
        self.Vt = np.append(self.Vt, (std_error ** 2) * (self.n * self.h))
        self.std_errors = np.append(self.std_errors, std_error)
        self.gps.loc[self.kept, str(trep[0])] = gps[self.kept]

    # 用户应该调用的唯一函数。 如果
    #如果使用了基函数，或要求对数据进行标准化处理，则将在此 # 函数中执行。
    # sdml 参数告诉我们是否要使用模拟的
    # dml 版本的估计器。
    def fit(self, X, T, Y, t_list, L=5, h=None, basis=False, standardize=False, sdml=False):
        self.mse_list = []  # 初始化 MSE 存储列表
        self.r2_list = []  # 初始化 R^2 存储列表
        self.reset()
        self.naive_count = 0
        self.n = len(Y)
        self.t_list = np.array(t_list, ndmin=1)
        self.L = L
        self.I_split = np.array_split(np.array(range(self.n)), L)

        # If no bandwidth is specified, use rule of thumb
        if h == None:
            self.h = np.std(T) * (self.n ** -0.2)
        else:
            self.h = h

        X, T, Y, t_list = self.reformat(X, T, Y, t_list, standardize)

        self.gps = pd.DataFrame(index=range(self.n))
        if basis == True:
            XT, Xf, ind = self.augment(X, T)
            if standardize == True:
                Xf = self.scale_non_dummies(Xf)[0]
                XT, scaler = self.scale_non_dummies(XT)
        else:
            XT = T
            Xf = X

        for t in np.array((t_list), ndmin=1):
            self.n = len(Y)
            trep = np.repeat(t, self.n)
            trep_sdml = np.array(random.choices(np.array(T) * self.h + t, k=self.n))
            if sdml == True:
                if basis == True:
                    Xt = self.augment(X, trep_sdml, ind)[0]
                    # XT,Xf_trash,ind = self.augment(X,trep_sdml)
                    if standardize == True:
                        # XT, scaler = self.scale_non_dummies(XT)
                        Xt = self.scale_non_dummies(Xt, scaler)[0]
                else:
                    Xt = trep_sdml
                    # XT = trep_sdml
            else:
                if basis == True:
                    Xt = self.augment(X, trep, ind)[0]
                    if standardize == True:
                        Xt = self.scale_non_dummies(Xt, scaler)[0]
                else:
                    Xt = trep
            self.fit_t(Xf, T, Y, trep, L, XT, Xt, trep_sdml, sdml)




        self.h_star = ((np.mean(self.Vt) / (4 * (np.mean(self.Bt ** 2)))) ** 0.2) * (self.n ** -0.2)

        if standardize == True:
            self.descale()

        self.gps.columns = self.t_list

    # 该函数通过添加基函数来增强数据。 Xf 是矩阵
    # 添加了仅取决于 X 的基函数的 X 矩阵，用于估算
    # GPS。 XT 是 X 和 T 以及附加基函数的矩阵，附加基函数可能取决于 # T 本身。
    # 本身取决于 T。 Xt 是矩阵 XT，但在 T=t 时求值。
    def augment(self, X, T, ind=None):
        T = T.reshape(len(T), 1)
        XT = np.column_stack((T, (T ** 2), (T ** 3), T * X))
        Xf = np.column_stack((X, X ** 2, X ** 3))
        Xf = np.unique(Xf, axis=1)
        if np.array_equal(ind, None):
            XT, ind = np.unique(XT, axis=1, return_index=True)
        else:
            XT = XT[:, ind]
        return XT, Xf, ind

    # 该函数用于缩放，但只对非虚拟变量进行缩放。
    # 重新缩放。
    def scale_non_dummies(self, D, scaler=None):
        D = pd.DataFrame(D)
        if scaler == None:
            scaler = sklearn.preprocessing.StandardScaler()
            D[D.select_dtypes('float64').columns] = scaler.fit_transform(D.select_dtypes('float64'))
        else:
            D[D.select_dtypes('float64').columns] = (D[D.select_dtypes(
                'float64').columns] - scaler.mean_) / scaler.scale_
        return np.array(D), scaler

    # 在拟合之前，该函数将确保所有数据和输入的格式正确无误
    # 在拟合之前。 数据按比例
    def reformat(self, X, T, Y, t_list, standardize):
        if standardize == True:
            df = pd.DataFrame(data=np.column_stack((Y, T, X)))
            self.scaling = {'mean_Y': np.mean(df[0]),
                            'sd_Y': np.std(df[0]),
                            'mean_T': np.mean(df[1]),
                            'sd_T': np.std(df[1])}
            df[df.select_dtypes('float64').columns] = sklearn.preprocessing.StandardScaler().fit_transform(
                df.select_dtypes('float64'))

            Y = df[0]
            T = df[1]
            X = df.loc[:, 2:]
            del df
            t_list = (t_list - self.scaling['mean_T']) / self.scaling['sd_T']
            self.h = self.h / self.scaling['sd_T']
        X = np.array((X))
        T = np.array((T))
        Y = np.array((Y))
        return X, T, Y, t_list

    # 该函数用于在估算结束时将估算值转换为
    # 可根据原始数据集的比例进行解释的数字
    def descale(self):
        self.std_errors = self.std_errors * self.scaling['sd_Y']
        self.h_star = self.h_star * self.scaling['sd_T']
        self.beta = (self.beta * self.scaling['sd_Y']) + self.scaling['mean_Y']
        self.h = self.h * self.scaling['sd_T']
        self.Vt = self.Vt * (self.scaling['sd_Y'] ** 2)


class NN_DDMLCT(DDMLCT):
    # 类似于 DDMLCT 的 "naive "函数，但经过调整以适合每个 t。
    def naive(self, Xf, XT, Xt, Y, I, I_C, L, K):
        gamma_model = self.model1.fit(Xf[I_C], Y[I_C], K[I_C])
        gamma = gamma_model.predict(Xf[I])
        return gamma

    # g 是内核平滑函数，用于对 GPS 进行估计。
    # 被建模。
    def ipw(self, Xf, g, I, I_C):
        self.model2.fit(Xf[I_C], g[I_C])
        gps = self.model2.predict(Xf[I])

        return gps

    # 对于给定的 t，调用此函数来估计每个子样本
    # 交叉拟合的结果。 如果 L=5，则每个 t 都要调用该函数 5 次。
    def fit_L(self, Xf, XT, Xt, Y, g, K, I, I_C, L):
        gamma = self.naive(Xf, XT, Xt, Y, I, I_C, L, K)
        gps = self.ipw(Xf, g, I, I_C)
        self.kept = np.concatenate((self.kept, I))

        # Compute the summand
        psi = np.mean(gamma) + np.mean(((Y[I] - gamma) * (K[I] / gps)))

        # Average over all indexes to get an estimate of beta hat
        beta_hat = np.mean(psi)

        return beta_hat, gamma, gps

    # 该函数会对给定拟合函数的 t 列表中的每一个 t 进行调用。
    # trep 是重复 n 次的 t 值。 XT 是
    # X、T 和添加的基函数的矩阵，这些基函数可能是 T 的函数。
    # Xf 是用于 GPS 估算的矩阵。
    # 用于 GPS 估算的矩阵。 如果使用附加基函数，则与 X 不同。
    # 它必须单独给出，否则
    # 否则就无法确定哪些变量取决于 T，哪些不取决于 T。
    def fit_t(self, Xf, T, Y, trep, L, XT, Xt, trep_sdml=None, sdml=False):
        self.kept = np.array((), dtype=int)
        mse_list = []  # 存储当前 t 值下所有子样本的 MSE
        r2_list = []  # 存储当前 t 值下所有子样本的 R^2
        # used for trimming which is not currently implemented

        T_t = T - trep
        g = gaussian_kernel(T_t, self.h)
        K = e_kernel(T_t, self.h)  #
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)

        # Iterate over all L sub-samples. I_split was defined in the fit function
        # so that the same split is used for all choice of t.
        for i in range(L):
            if L == 1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I = self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.n) if x not in I]

            beta_hat[i], gamma[I], gps[I] = self.fit_L(Xf, XT, Xt, Y, g, K, I, I_C, i)
            # 计算每折的 MSE 并存储
            mse = mean_squared_error(Y[I], gamma[I])
            mse_list.append(mse)
            r2 = r2_score(Y[I], gamma[I])
            r2_list.append(r2)
            beta_hat[i] = 16 * beta_hat[i]
            # 现在，我们对所有子样本进行平均，得出估计值和标准误差。
            # 计算平均 MSE
        avg_mse = np.mean(mse_list)
        avg_r2 = np.mean(r2_list)  # 当前 t 值的平均 R^2

        self.mse_list.append(avg_mse)  # 累积 MSE 到类属性中
        self.r2_list.append(avg_r2)

            # We now average over all sub-samples to get our estimates and standard
        # errors.
        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta, beta_hat)
        IF = (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept]) + gamma[self.kept] - beta_hat
        std_error = np.sqrt((1 / ((self.n) ** 2)) * np.sum(IF ** 2))

        self.Bt = np.append(self.Bt, (1 / (self.n * (self.h ** 2))) * (
            np.sum((K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept]))))
        self.Vt = np.append(self.Vt, (std_error ** 2) * (self.n * self.h))
        self.std_errors = np.append(self.std_errors, std_error)
        self.gps.loc[self.kept, str(trep[0])] = gps[self.kept]

    # This class is used to compute the DDMLCT estimator with alternative GPS estimation.


# we currently do not implement this for our numerical results due to computational
# infeasibility.
# optimization for later: Prevent computing of all g's every calling of ipw
# allow user input of t_grid and epsilon.
class DDMLCT_gps2(DDMLCT):
    def ipw(self, Xf, g, T, t, I, I_C):
        epsilon = 0.025
        t_grid = np.arange(t - 1.5 * self.h, t + 1.5 * self.h, self.h / 50)
        self.model2.fit(Xf[I_C], g[I_C])
        cdf_hat = self.model2.predict(Xf[I])
        # print(np.std(cdf_hat))
        cdf_hat = cdf_hat.reshape((len(cdf_hat), 1))
        cdf_hats = np.zeros((len(I), len(t_grid)))
        for i in range(len(t_grid)):
            trep = np.repeat(t_grid[i], self.n)
            t_T = trep - T
            g = norm.cdf(t_T / self.h)
            self.model2.fit(Xf[I_C], g[I_C])
            cdf_hats[:, i] = self.model2.predict(Xf[I])
        # find value of t with closest to cdf-hat-epsilon.
        # df = pd.DataFrame()
        # df['cdf'] = cdf_hats
        # df['cdf-eps'] = cdf_hat-epsilon
        # df['cdf+eps'] = cdf_hat+epsilon
        # print(df)
        # df.loc[df['cdf']>df['cdf-eps'],'cdf'].idxmin()
        # cdf_hats[cdf_hats<epsilon]=0
        # cdf_hat[cdf_hat<0]=0

        df = pd.DataFrame(cdf_hats)
        lower = df[df > (cdf_hat - epsilon)].T.apply(pd.Series.first_valid_index)

        lower = lower.fillna(0)
        lower = lower.astype(int)
        lower = lower.values.tolist()

        df = pd.DataFrame(cdf_hats)
        upper = df[df > (cdf_hat + epsilon)].T.apply(pd.Series.first_valid_index)

        upper = upper.fillna(len(t_grid) - 1)
        upper = upper.astype(int)
        upper = upper.values.tolist()
        # print(np.sum(np.array(upper)<np.array(lower)))
        # print(cdf_hat[(np.array(upper)<np.array(lower))][0])
        # print(cdf_hats[np.array(upper)<np.array(lower),:][0])
        # problem_cdf = cdf_hat[(np.array(upper)<np.array(lower))][0][0]
        # test = pd.Series(cdf_hats[np.array(upper)<np.array(lower),:][0])
        # print(test[test>(problem_cdf-epsilon)].first_valid_index,test[test>(problem_cdf+epsilon)].first_valid_index)
        # #print(cdf_hat)
        # lower = np.argmin(abs(cdf_hats-(cdf_hat-epsilon)), axis=1)
        # upper = np.argmin(abs(cdf_hats-(cdf_hat+epsilon)), axis=1)

        t_matrix = np.repeat(np.array(t_grid, ndmin=2), len(I), axis=0)

        t_upper = t_matrix[np.arange(len(t_matrix)), upper]
        t_lower = t_matrix[np.arange(len(t_matrix)), lower]

        inverse_gps = (t_upper - t_lower) / (2 * epsilon)
        # print(np.sum(inverse_gps==0))

        gps = 1 / inverse_gps
        return gps

    def fit_L(self, Xf, XT, Xt, Y, g, T, K, I, I_C, L, t):
        gamma = self.naive(Xf, XT, Xt, Y, I, I_C, L)
        gps = self.ipw(Xf, g, T, t, I, I_C)
        self.kept = np.concatenate((self.kept, I))
        Y_pred, residuals = self.residual_analysis(Xf, XT, Xt, Y, I, I_C, L)

        # Compute the summand
        psi = np.mean(gamma) + np.mean(((Y[I] - gamma) * (K[I] / gps)))

        # Average over all indexes to get an estimate of beta hat
        beta_hat = np.mean(psi)

        return beta_hat, gamma, gps,Y_pred, residuals

    def fit_t(self, Xf, T, Y, trep, L, XT, Xt, trep_sdml=None, sdml=False):

        self.kept = np.array((), dtype=int)  # used for trimming which is not currently implemented

        T_t = T - trep
        t_T = trep - T
        g = norm.cdf(t_T / self.h)
        K = e_kernel(T_t, self.h)  #
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)

        # Iterate over all L sub-samples. I_split was defined in the fit function
        # so that the same split is used for all choice of t.
        for i in range(L):
            if L == 1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I = self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.n) if x not in I]

            beta_hat[i], gamma[I], gps[I] = self.fit_L(Xf, XT, Xt, Y, g, T, K, I, I_C, i, trep[0])

            # We now average over all sub-samples to get our estimates and standard
        # errors.
        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta, beta_hat)
        IF = (K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept]) + gamma[self.kept] - beta_hat
        std_error = np.sqrt((1 / ((self.n) ** 2)) * np.sum(IF ** 2))
        self.Bt = np.append(self.Bt, (1 / (self.n * (self.h ** 2))) * (
            np.sum((K[self.kept] / gps[self.kept]) * (Y[self.kept] - gamma[self.kept]))))
        self.Vt = np.append(self.Vt, (std_error ** 2) * (self.n * self.h))
        self.std_errors = np.append(self.std_errors, std_error)
        self.gps.loc[self.kept, str(trep[0])] = gps[self.kept]


