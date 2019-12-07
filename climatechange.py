#!/usr/bin/env python
# coding: utf-8

# 气候数据分析

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()

# 导入数据

yearsBase, meanBase = np.loadtxt('./resources/5-year-mean-1951-1980.csv', delimiter=',', usecols=(0, 1), unpack=True)
years, mean = np.loadtxt('./resources/5-year-mean-1882-2014.csv', delimiter=',', usecols=(0, 1), unpack=True)

# 创建散点图

plt.scatter(yearsBase, meanBase)
plt.title('scatter plot of mean temp difference vs year')   # 平均温差与年份的散点图
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)     # 平均温差
plt.show()

# 使用 scikit-learn 执行线性回归

# 准备 Linear Regression 模型并实例化
model = LinearRegression(fit_intercept=True)

# 创建模型
model.fit(yearsBase[:, np.newaxis], meanBase)
mean_predicted = model.predict(yearsBase[:, np.newaxis])

# 生成一个图表
plt.scatter(yearsBase, meanBase)
plt.plot(yearsBase, mean_predicted)
plt.title('scatter plot of mean temp difference vs year')       # 平均温差与年份的散点图
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)     # 平均温差
plt.show()

print(' y = {0} * x + {1}'.format(model.coef_[0], model.intercept_))


# 使用 Seaborn 分析数据

plt.scatter(years, mean)
plt.title('scatter plot of mean temp difference vs year')  # 平均温差与年份的散点图
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)     # 平均温差
sns.regplot(yearsBase, meanBase)
plt.show()

