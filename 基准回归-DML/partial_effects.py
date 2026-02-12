# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:33:46 2020
Last update Monday Jan 10 1:13 pm 2022

该文件应在 empirical_application.py 之后运行。 其目的是
的主要估计值，并计算局部效应。
函数 "shift "用于创建一个贝塔向量，并将其移动指定的位数。
指定位数的贝塔向量，这样就可以很容易地计算出
近似给定贝塔附近的斜率。

"""

import numpy as np
import pandas as pd
import os

# 此函数对估计值向量进行平移，以便计算不同时间点t的估计值之间的差异。

def shift(xs, gap):
    e = np.empty_like(xs)
    if gap >= 0:
        e[:gap] = np.nan
        e[gap:] = xs[:-gap]
    else:
        e[gap:] = np.nan
        e[:gap] = xs[-gap:]
    return e

# 第一个循环计算最优带宽下的估计值。第二个循环计算经验法则带宽下的估计值。

#ml_list = ['lasso','rf','nn','knn']
#ml_list = ['lasso','nn','knn']
ml_list = ['nn']
gap = 1
eta = 2*gap
for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5_hstar.xlsx'
    file = path + name
    dat = pd.read_excel(file, engine='openpyxl')
    h = dat['h'][0]
    dat['partial effect'] = (shift(dat['beta'],-gap)-shift(dat['beta'],gap))/eta
    dat['se partial effect'] = ((np.sqrt(15/6)/h)*dat['se'])

    dat.to_excel(file,index=False)


for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5.xlsx'
    file = path + name
    dat = pd.read_excel(file, engine='openpyxl')
    h = dat['h'][0]
    dat['partial effect'] = (shift(dat['beta'],-gap)-shift(dat['beta'],gap))/eta
    dat['se partial effect'] = ((np.sqrt(15/6)/h)*dat['se'])


    dat.to_excel(file,index=False)











