import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from time import time
from sklearn.externals import joblib


# 加载.csv格式的数据
LOAD_PATH = "../data_set/dimension13_set"


def load_classify_data(load_path=LOAD_PATH):
    csv_path = os.path.join(load_path, "sample13_set_01.csv")
    return pd.read_csv(csv_path)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


if __name__ == "__main__":
    # 导入数据
    sample = load_classify_data()
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


    # 将13个维度单独取出，预测结果
    sample_data_13 = sample_data_prepared[:, :13]

    # 从硬盘上加载模型
    model = joblib.load(filename="model_predict.gz")
    result = model.predict(sample_data_13)
    result_label = pd.DataFrame(
        result,
        columns=["class_result"]
    )
    # print(result_label)

    for i in result_label.index:
        if result_label.iloc[i].bool():
            result_label.iloc[i] = "OK"
        else:
            result_label.iloc[i] = "NG"
    print(result_label)
    result_label.to_csv('result_label.csv', sep=',', header=True, index=True)