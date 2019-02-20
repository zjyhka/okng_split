from okng_class import data_load
from sklearn.pipeline import Pipeline
from okng_class.DataFrameSelector import DataFrameSelector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion


# 导入数据
sample = data_load.load_classify_data()

# 预处理数据
# 删除前四列无关属性
sample_data = sample.drop(["number", "parameter_01", "parameter_02(x_coordinate)",
                           "parameter_03(y_coordinate)"],
                          axis=1)

# 只保留数值属性
sample_data_num = sample_data.drop(["class(OK/NG)"], axis=1)
# 取出数值属性的列名
num_attribs = list(sample_data_num)
# 取出文本属性的
cat_attribs = ["class(OK/NG)"]

# 数据转换流水线，将维度13列数值进行标准化处理，最后一列属性不做处理
num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs)),
    ("std_scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs)),
])

full_pipeline = FeatureUnion(
    transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# 将数据输入到流水线中，得出准备好的数据
sample_data_prepared = full_pipeline.fit_transform(sample_data)

# 将最后一列分类标签单独取出，转换为一维数组
sample_data_label = sample_data_prepared[:, -1:]
sample_data_label = sample_data_label.flatten()
# 将标签列转化为布尔值 true为"OK" false为"NG" 便于后续衡量模型指标
label_train = (sample_data_label == "OK")

# 将13个维度单独取出，为训练做好准备
sample_data_13 = sample_data_prepared[:, :13]


# 导入数据
some_test = data_load.load_test_data()
# 删除前四列无关属性
some_test = some_test.drop(["number", "parameter_01", "parameter_02(x_coordinate)", "parameter_03(y_coordinate)"],
                          axis=1)

some_test_prepared = full_pipeline.fit_transform(some_test)
some_test_13 = some_test_prepared[:, :13]
some_test_label = some_test_prepared[:, -1:]
some_test_label = some_test_label.flatten()
# 将标签列转化为布尔值 true为"OK" false为"NG" 便于后续衡量模型指标
label_test = (some_test_label == "OK")


# 导入数据
some_test_02 = data_load.load_test02_data()
# 删除前四列无关属性
some_test_02 = some_test_02.drop(["number", "parameter_01", "parameter_02(x_coordinate)", "parameter_03(y_coordinate)"],
                          axis=1)

some_test02_prepared = full_pipeline.fit_transform(some_test_02)
some_test02_13 = some_test02_prepared[:, :13]
some_test02_label = some_test02_prepared[:, -1:]
some_test02_label = some_test02_label.flatten()


# 导入数据
some_test_03 = data_load.load_test03_data()
# 删除前四列无关属性
some_test_03 = some_test_03.drop(["number", "parameter_01", "parameter_02(x_coordinate)", "parameter_03(y_coordinate)"],
                          axis=1)

some_test03_prepared = full_pipeline.fit_transform(some_test_03)
some_test03_13 = some_test03_prepared[:, :13]
some_test03_label = some_test03_prepared[:, -1:]
some_test03_label = some_test03_label.flatten()
