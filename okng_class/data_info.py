from okng_class import data_load
import matplotlib.pyplot as plt


# 快速查看数据结构
sample_data = data_load.load_classify_data()
# 查看前5行，确认数据导入成功
print(sample_data.head())
print("-----------------------------")
# 查看样本数据信息，无空值
print(sample_data.info())
# 查看数值属性的摘要，输出到info_01.xls
print(sample_data.describe().to_excel("info_01.xls"))

# 简单绘制直方图查看
sample_data.hist(bins=50, figsize=(20, 15))
plt.show()






