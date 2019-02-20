from okng_class.classify_prepare import sample_data_13
from okng_class.classify_prepare import label_train
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from time import time
from sklearn.externals import joblib

# 使用LogisticRegression()训练
# For small datasets, ‘liblinear’ is a good choice
log_reg = LogisticRegression(C=1.0, solver='liblinear')
start = time()
log_reg.fit(sample_data_13, label_train)
print("train_spend_time =  %.2f seconds" % (time() - start))


if __name__ == "__main__":
    # 交叉验证分数，使用accuracy
    print("cross_val_score: ", cross_val_score(log_reg, sample_data_13, label_train,
                                               cv=5, scoring="accuracy"))

    # 返回交叉验证每个折叠的预测
    label_train_pred = cross_val_predict(log_reg, sample_data_13, label_train, cv=5)

    # precision_score
    print("precision_score = ", precision_score(label_train, label_train_pred))

    # recall_score
    print("recall_score = ", recall_score(label_train, label_train_pred))

    # F1_score
    print("F1_score = ", f1_score(label_train, label_train_pred))

    # save_model到硬盘
    joblib.dump(value=log_reg, filename="log_reg_model.gz", compress=True)
    print("model has saved")

    # 从硬盘上加载模型
    model = joblib.load(filename="log_reg_model.gz")
    result = model.predict(sample_data_13)
    print(result)


