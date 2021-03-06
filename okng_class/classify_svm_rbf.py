from okng_class.classify_prepare import sample_data_13
from okng_class.classify_prepare import label_train
from sklearn.svm import SVC
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
                "C": sp_randint(1, 10),
                "degree": sp_randint(1, 6),
                "gamma": sp_randint(0.1, 10)
}

# 使用svm核方法，核函数采用高斯核函数
svm_clf_rbf = SVC(kernel="rbf")
# run randomized search 查找最优参数
n_iter_search = 20
random_search = RandomizedSearchCV(svm_clf_rbf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)


start = time()
random_search.fit(sample_data_13, label_train)
print("spend_time = %.2f seconds" % (time() - start))
print("best_estimator", random_search.best_estimator_)

