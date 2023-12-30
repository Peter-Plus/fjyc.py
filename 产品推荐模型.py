from surprise import Reader, Dataset, SVD

# 加载数据集（假设您的数据集格式为：用户ID，商品ID，评分）
数据集 = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], Reader())

# 训练模型（这里使用 SVD 矩阵分解算法）
模型 = SVD()
模型.fit(数据集.build_full_trainset())

# 为特定用户进行商品推荐（例如，用户 ID 为 10 的用户）
用户ID = 10
用户未评分商品 = 数据集.build_anti_testset([用户ID])
预测评分 = 模型.test(用户未评分商品)

# 推荐评分最高的商品
推荐商品 = sorted(预测评分, key=lambda x: x.est, reverse=True)[:5]

print("为用户 ID 为 10 的用户推荐的商品：")
for 推荐项 in 推荐商品:
    print(f"商品ID：{推荐项.iid}, 预测评分：{推荐项.est:.2f}")
