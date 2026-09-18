# -*- coding: utf-8 -*-
# @Time    : 8/9/2024 4:13 pm
# @Author  : Wenyuan Li
# @File    : crop_yield_prediction.py
# @Description : Crop Yield Prediction with Enhanced Evaluation
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt

# 1. 加载数据
data_file = r'crop_yield/yield_data.csv'
# 加载数值数据
numerical_data = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols=(2, 3, 4, 5))
# 加载文本数据
text_data = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols=(0, 1), dtype='str')
# 加载目标数据
target_data = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols=(6))

# 2. 将文本数据转换为 One-Hot 编码
text_encoder = OneHotEncoder()
text_encoder.fit(text_data)
text_data_onehot = text_encoder.transform(text_data).toarray()
print("One-Hot 编码后数据维度:", text_data_onehot.shape)

# 3. 合并数据并划分训练集/测试集
input_data = np.concatenate((text_data_onehot, numerical_data), axis=1)
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, test_size=0.25)

# 4. 数据归一化
scaler = MinMaxScaler()
scaler.fit(train_input)  # 计算训练集的最小值和最大值
train_input = scaler.transform(train_input)  # 归一化训练集
test_input = scaler.transform(test_input)  # 归一化测试集

# 5. 建立并训练模型
yield_model = LinearRegression()
yield_model.fit(train_input, train_target)

# 6. 预测与评估 (测试集)
target_pred = yield_model.predict(test_input)
mse = mean_squared_error(test_target, target_pred)
rmse = np.sqrt(mse)  # 新增：计算 RMSE
r2 = r2_score(test_target, target_pred)
mae = mean_absolute_error(test_target, target_pred)

print('【测试集评估】MSE: {0:.2f}, RMSE: {1:.2f}, R2: {2:.2f}, MAE: {3:.2f}'.format(mse, rmse, r2, mae))

# 7. 绘制测试集散点图（含 1:1 线和指标）
plt.figure(figsize=(8, 6))
plt.scatter(test_target, target_pred, marker='o', s=20, alpha=0.5, label='Data points')

# 绘制 1:1 线（动态获取坐标轴范围）
min_val = min(test_target.min(), target_pred.min())
max_val = max(test_target.max(), target_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line') # 红色虚线

# 将评估指标显示在图上
text_x = min_val + (max_val - min_val) * 0.05
text_y = max_val - (max_val - min_val) * 0.15
plt.text(text_x, text_y, 'MSE: {0:.2f}'.format(mse), fontsize=12)
plt.text(text_x, text_y - (max_val - min_val) * 0.05, 'RMSE: {0:.2f}'.format(rmse), fontsize=12)
plt.text(text_x, text_y - (max_val - min_val) * 0.10, 'MAE: {0:.2f}'.format(mae), fontsize=12)
plt.text(text_x, text_y - (max_val - min_val) * 0.15, '$R^2$: {0:.2f}'.format(r2), fontsize=12)

plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Test Set: Crop Yield Prediction with 1:1 Line')
plt.legend()
plt.tight_layout()
plt.show()

# 8. 绘制测试集散点密度图（Hexbin）
plt.figure(figsize=(8, 6))
hb = plt.hexbin(test_target, target_pred, gridsize=30, cmap='Blues')
plt.colorbar(hb, label='Count')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Test Set: Scatter Density Plot (Hexbin)')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 评估训练集并绘制散点图（含 1:1 线和指标）
train_pred = yield_model.predict(train_input)
train_mse = mean_squared_error(train_target, train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(train_target, train_pred)
train_mae = mean_absolute_error(train_target, train_pred)

print('【训练集评估】MSE: {0:.2f}, RMSE: {1:.2f}, R2: {2:.2f}, MAE: {3:.2f}'.format(train_mse, train_rmse, train_r2, train_mae))

plt.figure(figsize=(8, 6))
plt.scatter(train_target, train_pred, marker='o', s=20, alpha=0.5, label='Data points')

# 绘制训练集 1:1 线
train_min_val = min(train_target.min(), train_pred.min())
train_max_val = max(train_target.max(), train_pred.max())
plt.plot([train_min_val, train_max_val], [train_min_val, train_max_val], 'r--', label='1:1 line')

# 自动定位并显示评估指标
train_text_x = train_min_val + (train_max_val - train_min_val) * 0.05
train_text_y = train_max_val - (train_max_val - train_min_val) * 0.15
plt.text(train_text_x, train_text_y, 'MSE: {0:.2f}'.format(train_mse), fontsize=12)
plt.text(train_text_x, train_text_y - (train_max_val - train_min_val) * 0.05, 'RMSE: {0:.2f}'.format(train_rmse), fontsize=12)
plt.text(train_text_x, train_text_y - (train_max_val - train_min_val) * 0.10, 'MAE: {0:.2f}'.format(train_mae), fontsize=12)
plt.text(train_text_x, train_text_y - (train_max_val - train_min_val) * 0.15, '$R^2$: {0:.2f}'.format(train_r2), fontsize=12)

plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Training Set: Crop Yield Prediction with 1:1 Line')
plt.legend()
plt.tight_layout()
plt.show()