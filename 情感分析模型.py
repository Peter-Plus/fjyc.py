from textblob import TextBlob

# 定义要分析的文本
文本 = "这部电影真是太棒了！演员演技精湛，剧情扣人心弦，画面也非常震撼。强烈推荐大家去看！"

# 创建 TextBlob 对象
分析结果 = TextBlob(文本)

# 获取情感极性分数（范围为 -1 到 1，表示从负面到正面）
情感分数 = 分析结果.sentiment.polarity

# 判断情感倾向
if 情感分数 > 0:
    情感倾向 = "正面"
elif 情感分数 < 0:
    情感倾向 = "负面"
else:
    情感倾向 = "中性"

print(f"文本情感分数：{情感分数:.2f}")
print(f"文本情感倾向：{情感倾向}")
