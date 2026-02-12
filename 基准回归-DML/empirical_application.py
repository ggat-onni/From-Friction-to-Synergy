# -*- coding: utf-8 -*-

# %% Import Necessary packages.
import Supplement
import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import matplotlib.pyplot as plt




# %% Read and Initialize Data
path = os.getcwd() + "\\Empirical Application\\"
directories = [path + "\\Estimates", path + "\\Estimates\\GPS"]
Supplement.make_dirs(directories)

name = '外部异质性2—知识产权保护_较弱.csv'
file = path + name
data = pd.read_csv(file, index_col=0)
data = data.sample(frac=1, random_state=20)

# 创建模型存储目录
model_dir = os.getcwd() + "\\Empirical Application\\Models\\"
os.makedirs(model_dir, exist_ok=True)

data = pd.concat([data.select_dtypes(exclude='int64'),
                  pd.get_dummies(data.select_dtypes('int64').astype('category'),
                                 drop_first=True)
                  ],
                 axis=1)

X = data.drop(['d', 'y'], axis=1)  # define covariate vector, excluding T and Y
T = data['d']  # define treatment vector
Y = data['y']  # define outcome vector

# %% Create the table of summary statistics
file = path + "\\Estimates\\Summary.xlsx"
summary_table = pd.DataFrame(index=pd.Index(['Share of Weeks Unemployed in Second Year (Y)',
                                             'Total Hours Spent in First Year Training (T)'],
                                            name='Variable'
                                            ),
                             columns=['Mean', 'Median', 'StdDev', 'Min', 'Max']
                             )
summary_table.iloc[0] = [np.mean(Y), np.median(Y), np.std(Y), np.min(Y), np.max(Y)]
summary_table.iloc[1] = [np.mean(T), np.median(T), np.std(T), np.min(T), np.max(T)]
summary_table.to_excel(file, index=True)

# %% Create the histogram of the treatment
plt.figure()
plt.title('Histogram of convergence', fontsize=16)
plt.xlabel('AI&GT convergence', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.hist(data[data['d'] < 1.8]['d'], bins=24, histtype='bar', ec='black', color='w')
plt.savefig(path + '\\Figures\\histogram.png')
plt.close()

# %% Define models and their parameters
args_lasso1 = {
    'alpha': 0.00069944,
    'max_iter': 10000,
    'tol': 0.0001,
    'normalize': True
}

args_lasso2 = {
    'alpha': 0.000160472,
    'max_iter': 10000,
    'tol': 0.0001,
    'normalize': True
}

model_lasso1 = linear_model.Lasso(**args_lasso1)
model_lasso2 = linear_model.Lasso(**args_lasso2)

model_nn1 = Supplement.NeuralNet1_emp_app(k=290,
                                          lr=0.15,
                                          momentum=0.9,
                                          epochs=100,
                                          weight_decay=0.05)

model_nn2 = Supplement.NeuralNet2_emp_app(k=289,
                                          lr=0.05,
                                          momentum=0.3,
                                          epochs=100,
                                          weight_decay=0.15)

model_knn1 = Supplement.NeuralNet1k_emp_app(k=302,
                                            lr=0.15,
                                            momentum=0.9,
                                            epochs=100,
                                            weight_decay=0.05)

model_knn2 = Supplement.NeuralNet2_emp_app(k=302,
                                           lr=0.05,
                                           momentum=0.3,
                                           epochs=100,
                                           weight_decay=0.15)

# Store all our models in a dictionary
models = {
    # 'lasso': [model_lasso1, model_lasso2],
    # 'rf': [model_rf1, model_rf2],
    'nn': [model_nn1, model_nn2],
    # 'knn': [model_knn1, model_knn2]
}

basis = {
    # 'lasso':True,
    # 'rf':False,
    'nn': False,
    # 'knn':False
}

# %% Iterate over all t and ml algorithms for estimation
t_list = np.arange(0, 3.75, 0.25)

# 基础带宽计算 (Rule of Thumb)
h_rot = np.std(T) * 3 * (len(Y) ** (-0.2))

# 【关键设置】
# h_main: 我们实际想要用来画图的带宽 (2倍经验带宽，曲线更平滑)
h_main = 2 * h_rot
# h_aux: 仅用于辅助计算 h_star 的带宽 (1倍经验带宽)
h_aux = h_rot
u = 0.5  # ratio between h_aux and h_main

L = 5
ml_list = ['nn']
col_names = ['t', 'beta', 'se', 'h_star', 'h']

for ml in ml_list:
    print(f"正在处理模型: {ml} ...")

    # 1. 拟合主要模型 (使用较宽的 h_main，结果较平滑)
    # 这是我们最终想要保存结果的模型
    print(f"  > 拟合主模型 (h = {h_main:.4f}) ...")
    if ml == 'knn':
        model_main = Supplement.NN_DDMLCT(models[ml][0], models[ml][1])
        model_main.fit(X, T, Y, t_list, L, h=h_main, basis=basis[ml], standardize=True)

        # 为了计算 h_star 填表，我们需要拟合辅助模型
        model_aux = Supplement.NN_DDMLCT(models[ml][0], models[ml][1])
        model_aux.fit(X, T, Y, t_list, L, h=h_aux, basis=basis[ml], standardize=True)
    else:
        model_main = Supplement.DDMLCT(models[ml][0], models[ml][1])
        model_main.fit(X, T, Y, t_list, L, h=h_main, basis=basis[ml], standardize=True)

        # 为了计算 h_star 填表，我们需要拟合辅助模型
        model_aux = Supplement.DDMLCT(models[ml][0], models[ml][1])
        model_aux.fit(X, T, Y, t_list, L, h=h_aux, basis=basis[ml], standardize=True)

    # 2. 计算 h_star (仅用于记录，不用于再次拟合)
    # 即使我们不用它画图，保留在 Excel 里展示“理论最优值”通常也是好的
    Bt = (model_main.beta - model_aux.beta) / ((model_main.h ** 2) * (1 - (u ** 2)))
    # 计算出的理论最优带宽
    calculated_h_star = np.mean(((model_aux.Vt / (4 * (Bt ** 2))) ** 0.2) * (model_main.n ** -0.2))

    print(f"  > 理论 h_star 为: {calculated_h_star:.4f} (仅作记录，不使用)")

    # 3. 输出结果
    # 注意：这里使用的是 model_main 的 beta (平滑结果)
    # h_star 列填入计算出的理论值，h 列填入实际使用的平滑带宽

    overall_mse = np.mean(model_main.mse_list)
    overall_r2 = np.mean(model_main.r2_list)
    print(f"  > 模型 {ml} 最终结果 (平滑版) - MSE: {overall_mse:.4f}, R^2: {overall_r2:.4f}")

    output = np.column_stack((np.array(t_list),
                              model_main.beta,  # 平滑的系数
                              model_main.std_errors,  # 平滑的标准误
                              np.repeat(calculated_h_star, len(t_list)),  # 记录理论值
                              np.repeat(model_main.h, len(t_list))))  # 记录实际值 (2*RoT)

    output = pd.DataFrame(output, columns=col_names)

    path_est = os.getcwd() + "\\Empirical Application\\Estimates\\"
    # 保持文件名简洁
    name_est = 'emp_app_' + str(ml) + '_c3_L5.xlsx'
    file_est = path_est + name_est
    output.to_excel(file_est)
    print(f"  > 结果已保存至: {name_est}")

    path_gps = os.getcwd() + "\\Empirical Application\\Estimates\\GPS\\"
    name_gps = 'GPS_' + str(ml) + '.xlsx'
    file_gps = path_gps + name_gps
    model_main.gps.to_excel(file_gps, index=True)

print("所有处理完成。")