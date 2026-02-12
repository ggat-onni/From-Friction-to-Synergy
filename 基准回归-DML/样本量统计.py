import pandas as pd
import numpy as np


# 读取原数据
data = pd.read_csv("Empirical Application/外部异质性1—数字基础设施_强.csv")  # 请根据实际情况修改文件路径

# 提取 d 列
d_data = data['d']

# 设定区间范围和区间大小
lower_bound = 0
upper_bound = 3.75
bin_size = 0.25

# 生成区间
bins = np.arange(lower_bound, upper_bound + bin_size, bin_size)

# 使用 pd.cut 来统计每个区间的数据数量
bin_counts = pd.cut(d_data, bins=bins, right=False).value_counts().sort_index()
# 只保留样本量，去掉区间标签
bin_counts_values = bin_counts.values
# 显示每个区间的样本量
print(bin_counts_values)
# 显示每个区间的样本量
print(bin_counts)