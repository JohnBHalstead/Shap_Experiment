# %%
# Packages, libraries, data, etc.

import time
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# models
knn_clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
rf_clf = RandomForestClassifier(n_estimators=200, max_features="auto", min_samples_split=10, min_samples_leaf=1,
                                max_depth=10, bootstrap=False, n_jobs=1)
nn_clf = MLPClassifier(hidden_layer_sizes=(15,), alpha=0.1, max_iter=100000)
ada_clf = AdaBoostClassifier(learning_rate=1.25, n_estimators=175)

# load and organize Wisconsin Breast Cancer Data 
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at the data
print(label_names)
print(labels)
print(feature_names)
print(features.shape)  # (569, 30)

# Random split data
X_tng, X_val, y_tng, y_val = train_test_split(features, labels, test_size=0.33, random_state=42)

print(X_tng.shape)  # (381, 30)
print(X_val.shape)  # (188, 30)

# %%
# Modeling
start = time.time()
for clf in (knn_clf, rf_clf, nn_clf, ada_clf):
    clf.fit(X_tng, y_tng)
    y_hat = clf.predict(X_val)
    print(clf.__class__.__name__, roc_auc_score(y_val, y_hat))

end = time.time()
print(end - start)

# KNeighborsClassifier 0.9345627235722215
# RandomForestClassifier 0.9386949549771803
# MLPClassifier 0.9776119402985074
# AdaBoostClassifier 0.9801406192179597

# Optimize and compare with Random Search
# AdaBoost
n_estimators = [int(x) for x in np.linspace(start=50, stop=200, num=25)]
learning_rate = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
AB_rgrid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
ABran = AdaBoostClassifier()
AB_RS = RandomizedSearchCV(ABran, param_distributions=AB_rgrid, scoring="roc_auc")
AB_RS.fit(X_tng, y_tng)
print(AB_RS.best_params_)
# {'n_estimators': 50, 'learning_rate': 1.3}

# Neural Network nn_clf = MLPClassifier(hidden_layer_sizes=(15,), alpha=0.1, max_iter=100000)
hidden_layer_sizes = ([int(x) for x in np.linspace(start=1, stop=51, num=2)],)
alpha = [0.01, 0.05, 0.1, 0.15, 0.2]
max_iter = [int(x) for x in np.linspace(start=10000, stop=200000, num=10000)]
NN_rgrid = {'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha, 'max_iter': max_iter}
NNran = MLPClassifier()
NN_RS = RandomizedSearchCV(NNran, param_distributions=NN_rgrid, scoring="roc_auc")
NN_RS.fit(X_tng, y_tng)
print(NN_RS.best_params_)

# {'max_iter': 24612, 'hidden_layer_sizes': [1, 51], 'alpha': 0.05}

# Random Forest, rf_clf = RandomForestClassifier(n_estimators=200, max_features="auto", min_samples_split=10, min_samples_leaf=1, max_depth=10, bootstrap=False, n_jobs=1)
n_estimators = [int(x) for x in np.linspace(start=500, stop=2000, num=10)]  # Number of trees in random forest
max_features = ['auto', 'sqrt']  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]  # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]  # Minimum number of samples required at each leaf node
bootstrap = [True, False]  # Method of selecting samples for training each tree

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

RFran = RandomForestClassifier()
RF_RS = RandomizedSearchCV(RFran, param_distributions=random_grid, scoring="roc_auc")
RF_RS.fit(X_tng, y_tng)
print(RF_RS.best_params_)

# {'n_estimators': 1666, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}

# K-nearest neighbor, knn_clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance')
n_neighbors = [int(x) for x in np.linspace(start=1, stop=20, num=1)]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size = [int(x) for x in np.linspace(start=10, stop=50, num=5)]
p = [1, 2]

KNN_rgrid = {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm, 'leaf_size': leaf_size, 'p': p}
KNNran = KNeighborsClassifier()
KNN_RS = RandomizedSearchCV(KNNran, param_distributions=KNN_rgrid, scoring="roc_auc")
KNN_RS.fit(X_tng, y_tng)
print(KNN_RS.best_params_)
# {'weights': 'uniform', 'p': 1, 'n_neighbors': 1, 'leaf_size': 40, 'algorithm': 'auto'}

# %%
# Running the optimized models
nn_clf = MLPClassifier(hidden_layer_sizes=(51,), alpha=0.05, max_iter=100000)
ada_clf = AdaBoostClassifier(learning_rate=1.3, n_estimators=50)
rf_clf = RandomForestClassifier(n_estimators=1666, max_features="auto", min_samples_split=2, min_samples_leaf=2,
                                max_depth=20, bootstrap=True, n_jobs=1)
knn_clf = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="auto", leaf_size=40, p=1)
# Modeling
start = time.time()
for clf in (nn_clf, ada_clf, rf_clf, knn_clf):
    clf.fit(X_tng, y_tng)
    y_hat = clf.predict(X_val)
    print(clf.__class__.__name__, roc_auc_score(y_val, y_hat))

end = time.time()
print(end - start)

# MLPClassifier 0.9585543357592204
# AdaBoostClassifier 0.9644134698408783
# RandomForestClassifier 0.9577525595164673
# KNeighborsClassifier 0.950289872949303

# use best results
nn_clf = MLPClassifier(hidden_layer_sizes=(15,), alpha=0.1, max_iter=100000)
ada_clf = AdaBoostClassifier(learning_rate=1.25, n_estimators=175)
rf_clf = RandomForestClassifier(n_estimators=1666, max_features="auto", min_samples_split=2, min_samples_leaf=2,
                                max_depth=20, bootstrap=True, n_jobs=1)
knn_clf = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="auto", leaf_size=40, p=1)
# Modeling
start = time.time()
for clf in (nn_clf, ada_clf, rf_clf, knn_clf):
    clf.fit(X_tng, y_tng)
    y_hat = clf.predict(X_val)
    print(clf.__class__.__name__, roc_auc_score(y_val, y_hat))

end = time.time()
print(end - start)

# MLPClassifier 0.9610830146786727
# AdaBoostClassifier 0.9801406192179597
# RandomForestClassifier 0.9577525595164673
# KNeighborsClassifier 0.950289872949303

# Applying SHAP to AdaBoost, which is an ensemble method
start = time.time()
explainADA = shap.TreeExplainer(ada_clf.predict, X_tng)
shap_values_ADA_val = explainADA.shap_values(X_val)
stop = time.time()
print(stop - start)
# time was 5953.293463230133 seconds

# build with the k-means summary as the feature set to speed process
X_tng_sum = shap.kmeans(X_tng, 10)
start = time.time()
explainADA1 = shap.KernelExplainer(ada_clf.predict,
                                   X_tng_sum)  # using the kmeans method switches the explainer from Tree to Kernel
shap_values_ADA_val1 = explainADA1.shap_values(X_val)
end = time.time()
print(end - start)
# time was 91.79060077667236 seconds

# Applying SHAP to Neural Network
start = time.time()
explainNN = shap.KernelExplainer(nn_clf.predict, X_tng)
shap_values_NN_val = explainNN.shap_values(X_val)
stop = time.time()
print(stop - start)
# time was 396.96149230003357 seconds

# build with the k-means summary as the feature set to speed process
start = time.time()
explainNN1 = shap.KernelExplainer(nn_clf.predict, X_tng_sum)
shap_values_NN_val1 = explainNN1.shap_values(X_val)
end = time.time()
print(end - start)
# time was 20.943408727645874 seconds

# Applying SHAP to Random Forest
start = time.time()
explainRF = shap.KernelExplainer(rf_clf.predict, X_tng)  # TreeExplainer doesn't work for Random Forest
shap_values_RF_val = explainRF.shap_values(X_val)
stop = time.time()
print(stop - start)
# time was 13862.432233095169 seconds

# build with the k-means summary as the feature set to speed process
start = time.time()
explainRF1 = shap.KernelExplainer(rf_clf.predict, shap.kmeans(X_tng, 10))
shap_values_RF_val1 = explainRF1.shap_values(X_val)
end = time.time()
print(end - start)
# time was 227.95396184921265 seconds

# Applying SHAP to KNN Classifier
start = time.time()
explainKNN = shap.KernelExplainer(knn_clf.predict, X_tng)  # TreeExplainer doesn't work for Random Forest
shap_values_KNN_val = explainKNN.shap_values(X_val)
stop = time.time()
print(stop - start)
# time was 3404.1099379062653 seconds

# build with the k-means summary as the feature set to speed process
start = time.time()
explainKNN1 = shap.KernelExplainer(knn_clf.predict, shap.kmeans(X_tng, 10))
shap_values_KNN_val1 = explainKNN1.shap_values(X_val)
end = time.time()
print(end - start)
# time was 126.0377447605133 seconds

# %%
# shap graphics
X_v = pd.DataFrame(X_val)
X_v.columns = [feature_names]

# Summary graphics
# Neural Networks Summary
shap.summary_plot(shap_values_NN_val, X_v, title="Neural Network Feature Summary", show=False)
plt.savefig('NN_org_sum.png', bbox_inches="tight", pad_inches=1)

shap.summary_plot(shap_values_NN_val1, X_v, title="Neural Network Feature Summary (Modified)", show=False)
plt.savefig('NN_mod_sum.png', bbox_inches="tight", pad_inches=1)

# Random Forest Summary
shap.summary_plot(shap_values_RF_val, X_v, title="Random Forest Feature Summary", show=False)
plt.savefig('RF_org_sum.png', bbox_inches="tight", pad_inches=1)

shap.summary_plot(shap_values_RF_val1, X_v, title="Random Forest Feature Summary (Modified)", show=False)
plt.savefig('RF_mod_sum.png', bbox_inches="tight", pad_inches=1)

# AdaBoost Summary
shap.summary_plot(shap_values_ADA_val, X_v, title="AdaBoost Feature Summary", show=False)
plt.savefig('ADA_org_sum.png', bbox_inches="tight", pad_inches=1)

shap.summary_plot(shap_values_ADA_val1, X_v, title="AdaBoost Feature Summary (Modified)", show=False)
plt.savefig('ADA_mod_sum.png', bbox_inches="tight", pad_inches=1)

# K Nearest Neighbor Summary
shap.summary_plot(shap_values_KNN_val, X_v, title="K-Nearest Neighbor Feature Summary", show=False)
plt.savefig('KNN_org_sum.png', bbox_inches="tight", pad_inches=1)

shap.summary_plot(shap_values_KNN_val1, X_v, title="K-Nearest Neighbor Feature Summary (Modified)", show=False)
plt.savefig('KNN_mod_sum.png', bbox_inches="tight", pad_inches=1)

#  Dependence graphics
shap.dependence_plot("rank(1)", shap_values_NN_val1, X_v, title="Top Dependent Features in Neural Network")

shap.dependence_plot("rank(1)", shap_values_KNN_val1, X_v, title="Top Dependent Features in K-Nearest Neighbor")

shap.dependence_plot("rank(1)", shap_values_RF_val1, X_v, title="Top Dependent Features in Random Forest")

shap.dependence_plot("rank(1)", shap_values_ADA_val1, X_v, title="Top Dependent Features in AdaBoost")

# Waterfall plots are disabled from the package
# Force Plots of single observations
shap.force_plot(explainNN.expected_value, shap_values_NN_val[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainNN1.expected_value, shap_values_NN_val1[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainRF.expected_value, shap_values_RF_val[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainRF1.expected_value, shap_values_RF_val1[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainKNN.expected_value, shap_values_KNN_val[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainKNN1.expected_value, shap_values_KNN_val1[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainADA.expected_value, shap_values_ADA_val[0], feature_names=feature_names, matplotlib=True)

shap.force_plot(explainNN1.expected_value, shap_values_ADA_val1[0], feature_names=feature_names, matplotlib=True)

# force plot on all observations
shap.force_plot(explainNN.expected_value, shap_values_NN_val, X_val, matplotlib=True) # matplotlib = True is not yet supported for force plots with multiple samples!

# Summary Bar
shap.summary_plot(shap_values_NN_val, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_NN_val1, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_RF_val, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_RF_val1, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_KNN_val, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_KNN_val1, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_ADA_val, X_val, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values_ADA_val1, X_val, feature_names=feature_names, plot_type="bar")

# Decision plots
shap.decision_plot(explainNN.expected_value, shap_values_NN_val, feature_names, link='logit')

shap.decision_plot(explainNN1.expected_value, shap_values_NN_val1, feature_names, link='logit')

shap.decision_plot(explainRF.expected_value, shap_values_RF_val, feature_names, link='logit')

shap.decision_plot(explainRF1.expected_value, shap_values_RF_val1, feature_names, link='logit')

shap.decision_plot(explainKNN.expected_value, shap_values_KNN_val, feature_names, link='logit')

shap.decision_plot(explainKNN1.expected_value, shap_values_KNN_val1, feature_names, link='logit')

shap.decision_plot(explainADA.expected_value, shap_values_ADA_val, feature_names, link='logit')

shap.decision_plot(explainADA1.expected_value, shap_values_ADA_val1, feature_names, link='logit')

# %%
# write shap files to drive
NN_shap_values = pd.DataFrame(shap_values_NN_val)
NN_shap_values.columns = [feature_names]
NN1_shap_values = pd.DataFrame(shap_values_NN_val1)
NN1_shap_values.columns = [feature_names]
RF_shap_values = pd.DataFrame(shap_values_RF_val)
RF_shap_values.columns = [feature_names]
RF1_shap_values = pd.DataFrame(shap_values_RF_val1)
RF1_shap_values.columns = [feature_names]
KNN_shap_values = pd.DataFrame(shap_values_KNN_val)
KNN_shap_values.columns =[feature_names]
KNN1_shap_values = pd.DataFrame(shap_values_KNN_val1)
KNN1_shap_values.columns = [feature_names]
ADA_shap_values = pd.DataFrame(shap_values_ADA_val)
ADA_shap_values.columns = [feature_names]
ADA1_shap_values = pd.DataFrame(shap_values_ADA_val1)
ADA1_shap_values.columns = [feature_names]

export1_csv = NN_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/NNshapValues.csv", header=True)
export2_csv = NN1_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/NN1shapValues.csv", header=True)
export3_csv = RF_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/RFshapValues.csv", header=True)
export4_csv = RF1_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/RFshapValues.csv", header=True)
export5_csv = KNN_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/KNNshapValues.csv", header=True)
export6_csv = KNN1_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/KNN1shapValues.csv", header=True)
export1_csv = ADA_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/ADAshapValues.csv", header=True)
export2_csv = ADA1_shap_values.to_csv (r"/Users/jhalstead/Documents/data/project2Out/ADA1shapValues.csv", header=True)

