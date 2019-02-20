from okng_class.classify_randomforest_best import rnd_clf
from okng_class.class_svm_rbf_best import svm_clf_rbf
from okng_class.classify_logistic import log_reg
from okng_class.classify_svm_poly_best import svm_clf_poly
from okng_class.classify_prepare import some_test_13
from okng_class.classify_prepare import label_test
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict


def plot_roc_curve(fpr, tpr, label=None, color=None):
    plt.plot(fpr, tpr, linewidth=2, label=label, color=color)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


# 使用测试集，得出各模型的交叉验证预测效果，计算出ROC AUC
# 逻辑斯蒂回归预测效果
log_reg_pre_scores = cross_val_predict(log_reg, some_test_13, label_test, cv=3,
                                       method="decision_function")
roc_auc_log = roc_auc_score(label_test, log_reg_pre_scores)
print("逻辑斯蒂回归的ROC AUC为", roc_auc_log)

# 随机森林预测效果
rnd_clf_pre_probas = cross_val_predict(rnd_clf, some_test_13, label_test, cv=3,
                                       method="predict_proba")
rnd_clf_pre_scores = rnd_clf_pre_probas[:, 1]  # score = proba of positive class
roc_auc_rnd = roc_auc_score(label_test, rnd_clf_pre_scores)
print("随机森林的ROC AUC为", roc_auc_rnd)

# 高斯rbf核函数SVM支持向量机预测效果
svm_rbf_pre_scores = cross_val_predict(svm_clf_rbf, some_test_13, label_test, cv=3,
                                       method="decision_function")
roc_auc_rbf = roc_auc_score(label_test, svm_rbf_pre_scores)
print("高斯rbf核函数支持向量机的ROC AUC为", roc_auc_rbf)

# 多项式poly核函数SVM支持向量机预测效果
svm_poly_pre_scores = cross_val_predict(svm_clf_poly, some_test_13, label_test, cv=3,
                                        method="decision_function")
roc_auc_poly = roc_auc_score(label_test, svm_poly_pre_scores)
print("高斯poly核函数支持向量机的ROC AUC为", roc_auc_poly)

fpr_log, tpr_log, thresholds_log = roc_curve(label_test, log_reg_pre_scores)
fpr_forest, tpr_forest, thresholds_forest = roc_curve(label_test, rnd_clf_pre_scores)
fpr_rbf, tpr_rbf, thresholds_rbf = roc_curve(label_test, svm_rbf_pre_scores)
fpr_poly, tpr_poly, thresholds_poly = roc_curve(label_test, svm_poly_pre_scores)

# 可视化ROC曲线
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr_log, tpr_log, "Logistic Regression", "red")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest", "blue")
plot_roc_curve(fpr_rbf, tpr_rbf, "SVM_rbf", "yellow")
plot_roc_curve(fpr_poly, tpr_poly, "SVM_poly", "green")
plt.legend(loc="lower right", fontsize=16)
plt.show()


