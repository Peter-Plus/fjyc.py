import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载故障预测数据集（假设您的数据集包含机器状态数据和是否发生故障的标签）
数据集 = pd.read_csv("故障预测数据.csv")

# 分割特征和标签
特征 = 数据集.drop("是否故障", axis=1)  # 所有列为特征，排除 "是否故障" 列
标签 = 数据集["是否故障"]

# 分割训练集和测试集
训练集特征, 测试集特征, 训练集标签, 测试集标签 = train_test_split(特征, 标签, test_size=0.2, random_state=42)

# 创建随机森林分类模型
模型 = RandomForestClassifier(n_estimators=100)  # 设置 100 棵树

# 训练模型
模型.fit(训练集特征, 训练集标签)

# 进行预测
预测值 = 模型.predict(测试集特征)

# 评估模型准确率
准确率 = accuracy_score(测试集标签, 预测值)
print(f"模型准确率：{准确率:.2f}")
