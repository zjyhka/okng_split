from okng_class.classify_prepare import sample_data_13
from okng_class.classify_prepare import label_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint

# 随机参数的搜索范围
param_dist = {
                "n_estimators": sp_randint(100, 500),
                "max_features": sp_randint(1, 13),
                "min_samples_split": sp_randint(2, 11),
                "max_leaf_nodes": sp_randint(10, 20),
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"]
}
# 随机森林模型
rnd_clf = RandomForestClassifier(n_jobs=-1)
# run randomized search 查找最优参数
n_iter_search = 20
random_search = RandomizedSearchCV(rnd_clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)


start = time()
random_search.fit(sample_data_13, label_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print("best_estimator", random_search.best_estimator_)


