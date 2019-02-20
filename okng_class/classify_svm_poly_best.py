from okng_class.classify_prepare import sample_data_13
from okng_class.classify_prepare import label_train
from sklearn.svm import SVC
from time import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict

# 使用SVM训练模型，多项式poly核函数
svm_clf_poly = SVC(kernel="rbf", C=4, coef0=1, gamma="scale")

start = time()
svm_clf_poly.fit(sample_data_13, label_train)
print("train_spend_time =  %.2f seconds" % (time() - start))

if __name__ == "__main__":
    # 交叉验证分数，使用accuracy
    print("cross_val_score: ", cross_val_score(svm_clf_poly, sample_data_13, label_train,
                                               cv=5, scoring="accuracy"))

    # 返回交叉验证每个折叠的预测
    label_train_pred = cross_val_predict(svm_clf_poly, sample_data_13, label_train, cv=5)

    # precision_score
    print("precision_score = ", precision_score(label_train, label_train_pred))

    # recall_score
    print("recall_score = ", recall_score(label_train, label_train_pred))

    # F1_score
    print("F1_score = ", f1_score(label_train, label_train_pred))
