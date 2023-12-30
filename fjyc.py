import pandas as pd  # 数据处理库
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import mean_squared_error  # 评估指标

# 加载房价数据集
数据 = pd.read_csv("房价数据.csv")

# 分割特征和标签
特征 = 数据[["面积", "卧室数量"]]  # 选择相关特征
标签 = 数据["价格"]

# 分割数据集为训练集和测试集
训练集特征, 测试集特征, 训练集标签, 测试集标签 = train_test_split(特征, 标签, test_size=0.2, random_state=42)

# 创建线性回归模型
模型 = LinearRegression()

# 训练模型
模型.fit(训练集特征, 训练集标签)

# 使用模型进行房价预测
预测值 = 模型.predict(测试集特征)

# 评估模型的准确性
误差 = mean_squared_error(测试集标签, 预测值)
评估结果 = f"模型的平均误差为：{误差:.2f}"

print(评估结果)

