import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载欺诈检测数据集（假设您的数据集包含交易信息和标签，其中标签为 0 表示正常交易，1 表示欺诈交易）
数据集 = pd.read_csv("欺诈检测数据.csv")

# 分割特征和标签
特征 = 数据集.drop("标签", axis=1)  # 所有列为特征，排除 "标签" 列
标签 = 数据集["标签"]

# 分割训练集和测试集
训练集特征, 测试集特征, 训练集标签, 测试集标签 = train_test_split(特征, 标签, test_size=0.2, random_state=42)

# 创建逻辑回归模型
模型 = LogisticRegression()

# 训练模型
模型.fit(训练集特征, 训练集标签)

# 进行预测
预测值 = 模型.predict(测试集特征)

# 评估模型准确率
准确率 = accuracy_score(测试集标签, 预测值)
print(f"模型准确率：{准确率:.2f}")
