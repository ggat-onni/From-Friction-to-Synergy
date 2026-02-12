# Double Debiased Machine Learning for Mediation Analysis with Continuous Treatments

This is the code for our [DML mediation paper with continuous treatments.](https://arxiv.org/abs/2503.06156)


## 1.Environment Dependencies
This project is developed under **Python 3.9**.
To set up the environment, use the following command:
```
pip install -r requirements.txt
```
## 2. Mediation Effect Experiment Commands
```
python test.py --run_multi_d --data_path "非数字企业智能化中介.csv" --t_col "t" --m_col "m" --y_col "y" --categorical_cols "id,year" --estimator kme_dml 
python test.py --run_multi_d --data_path "数字企业绿色化中介.csv" --t_col "t" --m_col "m" --y_col "y" --categorical_cols "id,year" --estimator kme_dml 
```


Reference:

A large part of the conditional density estimation code comes from the library
https://github.com/freelunchtheorem/Conditional_Density_Estimation. 
