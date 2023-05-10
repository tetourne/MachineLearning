import sys
import logging
import joblib
import numpy as np
import scipy as sp
from pathlib import Path
from MachineLearning import functions as fn
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


do_cross_val = False
do_grid_search = False


########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / 'mnist.log'
logger = fn.my_logger(logs_file)


########## Load data ##########
logger.info("Loading MNIST dataset")
mnist = fn.load_mnist()
X = mnist.data
y = mnist.target
logger.warning("test")
stop
# split train and test sets
train_length = 60000  # 60000 for the full MNIST dataset
X_train, X_test = X[:train_length], X[train_length:]
y_train, y_test = y[:train_length], y[train_length:]
# shuffle the training set to have similiar cross-validation folds
np.random.seed(42)
shuffle_index = np.random.permutation(train_length)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# select only a subset of the training set
subset_len = 15000
X_train = X_train[:subset_len]
y_train = y_train[:subset_len]

# # build labels for each digit
# digits = [str(i) for i in np.arange(10)]
# y_train_bool = [(y_train == i) for i in digits]
# y_train_bool = np.array(y_train_bool)


########## Train Models ##########
# Use SVM classifier
svm_clf = Pipeline([
    ('standardscaler', StandardScaler()),  # necessary for SVM classifier
    ('svc', SVC(kernel='poly', degree=3, coef0=1, C=5))
    # LinearSVC()
])
# logger.info("Training SVM classifier")
# svm_clf.fit(X_train, y_train)

########## Cross Validation ##########
if do_cross_val:
    ### Accuracy:
    # how much good predictions have been done
    accuracies = cross_val_score(svm_clf, X_train, y_train, cv=3, scoring="accuracy")
    logger.info("Accucaries: {}".format(accuracies))

    ### Confusion Matrix:
    # [[TN, FP],[FN, TP]]
    y_train_pred = cross_val_predict(svm_clf, X_train, y_train, cv=3)
    cm = confusion_matrix(y_train, y_train_pred)
    logger.info("Confusion matrix:")
    logger.info(cm)

    ### Precision and recall
    # preci: TP/(TP+FP) : ratio of good positive predictions
    # recall or sensitivity of true positive rate: TP/(TP+FN) : ratio of positive instances corretly detected by the classifier
    preci = precision_score(y_train, y_train_pred, average='macro')
    logger.info("Precision: {}".format(preci))
    recall = recall_score(y_train, y_train_pred, average='macro')
    logger.info("Recall: {}".format(recall))
    f1score = f1_score(y_train, y_train_pred, average='macro')
    logger.info("F1 score: {}".format(f1score))


########## Grid Search ##########
svm_clf = Pipeline([
    ('standardscaler', StandardScaler()),  # necessary for SVM classifier
    ('svc', SVC())
])
param_grid = [
    {'svc__kernel': ['poly'], 'svc__gamma': [1e-2, 1, 10], 'svc__C': [0.1, 1, 10]}
]
grid_search = GridSearchCV(svm_clf, param_grid, cv=3)
if do_grid_search:
    logger.info("Doing GridSearchCV with:")
    logger.info(param_grid)
    grid_search.fit(X_train, y_train)
    # Save search
    pkl_path = Path(__file__).parents[0] / "pkls"
    if not pkl_path.is_dir():
        pkl_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search, pkl_path / "grid_search_svc_clf_2.pkl")
if not do_grid_search:
    pkl_path = Path(__file__).parents[0] / "pkls"
    logger.info("Loading GridSearchCV:")
    grid_search = joblib.load(pkl_path / "grid_search_svc_clf.pkl")
logger.info(grid_search.best_params_)
logger.info(grid_search.best_estimator_)
fn.display_cv_results(grid_search)


########## Test best model ##########
# Define best model
final_model = Pipeline([
    ('standardscaler', StandardScaler()),
    ('svc', SVC(C=0.1, gamma=1, kernel='poly'))
])

# Read again full data (used subset to do grid_search)
X_train, X_test = X[:train_length], X[train_length:]
y_train, y_test = y[:train_length], y[train_length:]
# shuffle the training set to have similiar cross-validation folds
np.random.seed(42)
shuffle_index = np.random.permutation(train_length)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

### Accuracy:
# how much good predictions have been done
cv = 5
logger.info("Training final model on {} cross validation folds".format(cv))
accuracies = cross_val_score(final_model, X_train, y_train, cv=cv, scoring="accuracy")
logger.info("Accuracies: {}".format(accuracies))

### Confusion Matrix:
# [[TN, FP],[FN, TP]]
y_train_pred = cross_val_predict(final_model, X_train, y_train, cv=cv)
cm = confusion_matrix(y_train, y_train_pred)
logger.info("Confusion matrix:")
logger.info(cm)

### Precision and recall
# preci: TP/(TP+FP) : ratio of good positive predictions
# recall or sensitivity of true positive rate: TP/(TP+FN) : ratio of positive instances corretly detected by the classifier
preci = precision_score(y_train, y_train_pred, average='macro')
logger.info("Precision: {}".format(preci))
recall = recall_score(y_train, y_train_pred, average='macro')
logger.info("Recall: {}".format(recall))
f1score = f1_score(y_train, y_train_pred, average='macro')
logger.info("F1 score: {}".format(f1score))

### Test on the test set
logger.info("Fitting final model on the full training set...")
final_model.fit(X_train, y_train)
logger.info("Testing on the test set:")
y_test_pred = final_model.predict(X_test)
preci = precision_score(y_test, y_test_pred, average='macro')
logger.info("Precision: {}".format(preci))
recall = recall_score(y_test, y_test_pred, average='macro')
logger.info("Recall: {}".format(recall))
f1score = f1_score(y_test, y_test_pred, average='macro')
logger.info("F1 score: {}".format(f1score))



### First try with explicitly using 10 classifiers....
# ########## Train Models ##########
# ### 10 binary classifiers (One vs all strategy)
# base_model = make_pipeline(
#     StandardScaler(),  # necessary for SVM classifiers
#     SVC(kernel='poly', degree=3, coef0=1, C=5)
#     # LinearSVC()
#     )
# svm_clfs = [clone(base_model) for i in digits]
# for i, mod, lab in zip(digits, svm_clfs, y_train_bool):
#     logger.info("Training SVM classifier for digit: {}".format(i))
#     mod.fit(X_train, lab)

# ### Check prediction
# j = 1
# logger.info("for digit {}, classifiers give:".format(y_train[j]))
# for i, mod in zip(digits, svm_clfs):
#     logger.info("{}: {}".format(i, mod.predict(X_train[j].reshape(1,-1))))


# ########## Cross Validation ##########
# accuracies_list, cm_list, preci_list, recall_list, f1score_list = [], [], [], [], []
# for i, mod, lab in zip(digits, svm_clfs, y_train_bool):
#     logger.info("Cross validation for digit: {}".format(i))
#     ### Accuracy:
#     # how much good predictions have been done
#     accuracies = cross_val_score(mod, X_train, lab, cv=3, scoring="accuracy")
#     logger.info("Accucaries: {}".format(accuracies))
#     accuracies_list.append(accuracies)
    
#     ### Confusion Matrix:
#     # [[TN, FP],[FN, TP]]
#     y_train_pred = cross_val_predict(mod, X_train, lab, cv=3)
#     cm = confusion_matrix(lab, y_train_pred)
#     logger.info("Confusion matrix:")
#     logger.info(cm)
#     cm_list.append(cm)

#     ### Precision and recall
#     # preci: TP/(TP+FP) : ratio of good positive predictions
#     # recall or sensitivity of true positive rate: TP/(TP+FN) : ratio of positive instances corretly detected by the classifier
#     preci = precision_score(lab, y_train_pred)
#     logger.info("Precision: {}".format(preci))
#     preci_list.append(preci)
#     recall = recall_score(lab, y_train_pred)
#     logger.info("Recall: {}".format(recall))
#     recall_list.append(recall)
#     f1score = f1_score(lab, y_train_pred)
#     logger.info("F1 score: {}".format(f1score))
#     f1score_list.append(f1score)

# logger.info("Averaged values are:")
# logger.info("- precision = {}".format(np.mean(preci_list)))
# logger.info("- recall = {}".format(np.mean(recall_list)))
# logger.info("- f1score = {}".format(np.mean(f1score_list)))


# ########## Grid Search ##########
# param_grid = [
#     {'kernel': ['poly', 'rbf'], 'gamma': [1e-10, 1e-7, 1e-4, 1], C: [0.1, 1, 10]}
# ]
# grid_search = GridSearchCV(kneighbors_clf, param_grid, cv=5)
# if do_grid_search:
#     logger.info("Doing GridSearchCV with:")
#     logger.info(param_grid)
#     grid_search.fit(X_train, y_train)
#     # Save search
#     pkl_path = Path(__file__).parents[0] / "pkls"
#     if not pkl_path.is_dir():
#         pkl_path.mkdir(parents=True, exist_ok=True)
#     joblib.dump(grid_search, pkl_path / "grid_search_kneighbors_clf.pkl")
# if not do_grid_search:
#     pkl_path = Path(__file__).parents[0] / "pkls"
#     logger.info("Loading GridSearchCV:")
#     grid_search = joblib.load(pkl_path / "grid_search_kneighbors_clf.pkl")
# grid_search.best_params_
# grid_search.best_estimator_
# fn.display_cv_results(grid_search)

