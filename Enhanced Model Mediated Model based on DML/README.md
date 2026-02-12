# Double Debiased Machine Learning for Mediation Analysis with Continuous Treatments

This is the code for our [DML mediation paper with continuous treatments.](https://arxiv.org/abs/2503.06156)

Please cite our work if you find it useful for your research and work:
```
@inproceedings{
zenati2025double,
title={Double Debiased Machine Learning for Mediation Analysis with Continuous Treatments},
author={Houssam Zenati and Judith Ab{\'e}cassis and Julie Josse and Bertrand Thirion},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
}
```

## 1. 环境依赖 (Dependencies)
本项目基于 **Python 3.9** 环境开发。
使用下面命令进行环境安装
```
pip install -r requirements.txt
```
## 2. 中介效应实验命令
```
python test.py --run_multi_d --data_path "非数字企业智能化中介.csv" --t_col "t" --m_col "m" --y_col "y" --categorical_cols "id,year" --estimator kme_dml 
python test.py --run_multi_d --data_path "数字企业绿色化中介.csv" --t_col "t" --m_col "m" --y_col "y" --categorical_cols "id,year" --estimator kme_dml 
```


Reference:

A large part of the conditional density estimation code comes from the library
https://github.com/freelunchtheorem/Conditional_Density_Estimation. 
