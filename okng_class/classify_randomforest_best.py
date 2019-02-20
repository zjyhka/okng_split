from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from okng_class.classify_prepare import sample_data_13
from okng_class.classify_prepare import label_train
from sklearn.ensemble import RandomForestClassifier
from time import time

# 随机森林模型
rnd_clf = RandomForestClassifier(bootstrap=True, criterion="entropy",
                                 max_features=5, max_leaf_nodes=14,
                                 n_estimators=309, n_jobs=-1, random_state=42)

start = time()
rnd_clf.fit(sample_data_13, label_train)
print("train_spend_time =  %.2f seconds" % (time() - start))

if __name__ == "__main__":
    # 交叉验证分数，使用accuracy
    print("cross_val_score: ", cross_val_score(rnd_clf, sample_data_13, label_train,
                                               cv=5, scoring="accuracy"))

    # 返回交叉验证每个折叠的预测
    label_train_pred = cross_val_predict(rnd_clf, sample_data_13, label_train, cv=5)

    # precision_score
    print("precision_score = ", precision_score(label_train, label_train_pred))

    # recall_score
    print("recall_score = ", recall_score(label_train, label_train_pred))

    # F1_score
    print("F1_score = ", f1_score(label_train, label_train_pred))
