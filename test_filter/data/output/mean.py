# 打开文件并读取内容
with open('/home/q661086/output/feature_scores.txt', 'r') as file:
    lines = file.readlines()
 
# 初始化变量
total_score = 0.0
feature_count = 0
 
# 遍历每一行，查找并提取 `Total Weighted Feature Score`
for line in lines:
    if "Total Weighted Feature Score" in line:
        # 提取score值
        score = float(line.split(':')[-1].strip())
        total_score += score
        feature_count += 1
 
# 计算平均值
average_score = total_score / feature_count if feature_count > 0 else 0
 
# 输出平均值
print(f"所有feature的Total Weighted Feature Score的平均值为: {average_score}")