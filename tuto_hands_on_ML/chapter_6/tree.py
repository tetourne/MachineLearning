import sys
import logging
import joblib
import numpy as np
import scipy as sp
from pathlib import Path
from scipy.stats import loguniform, uniform, randint
from matplotlib import pyplot as plt
from MachineLearning import functions as fn
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


plot_dataset = False
do_cross_val = False
do_rnd_search = True


########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / 'tree.log'
logger = fn.my_logger(logs_file)


########## Load dataset ##########
# generate dataset
X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons,
                                                    test_size=0.2,
                                                    random_state=42)
# plot dataset
if plot_dataset:
    plt.plot(X_train[:,0][y_train==0], X_train[:,1][y_train==0], 'bs', alpha=0.3)
    plt.plot(X_train[:,0][y_train==1], X_train[:,1][y_train==1], 'g^', alpha=0.3)
    plt.grid()
    plt.show()


########## Tune model ##########
tree_clf = DecisionTreeClassifier(random_state=42)
mod_name = 'tree_clf'

### Cross Validation
# I get bad results because I guess the model is free, and then it overfits the training
# data and generalize very poorly
if do_cross_val:
    cv = 5
    ### Accuracy:
    # how much good predictions have been done
    accuracies = cross_val_score(tree_clf, X_train, y_train, cv=cv, scoring="accuracy")
    logger.info("Accucaries: {}".format(accuracies))

    ### Confusion Matrix:
    # [[TN, FP],[FN, TP]]
    y_train_pred = cross_val_predict(tree_clf, X_train, y_train, cv=cv)
    cm = confusion_matrix(y_train, y_train_pred)
    logger.info("Confusion matrix:")
    logger.info(cm)

    ### Precision and recall
    # preci: TP/(TP+FP) : ratio of good positive predictions
    # recall or sensitivity of true positive rate: TP/(TP+FN) : ratio of positive instances corretly detected by the classifier
    preci = precision_score(y_train, y_train_pred)
    logger.info("Precision: {}".format(preci))
    recall = recall_score(y_train, y_train_pred)
    logger.info("Recall: {}".format(recall))
    f1score = f1_score(y_train, y_train_pred)
    logger.info("F1 score: {}".format(f1score))


### Randomized Search
tree_clf = DecisionTreeClassifier(random_state=42)
mod_name = 'tree_clf'
param_distrib = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': randint(1,50),
    'min_samples_split': randint(1, 100),
    'min_samples_leaf': randint(1, 100),
    'max_leaf_nodes': randint(1,100)
}
# scoring methods can be accuracy, precision, recall, f1score, roc_auc...
# By default, it uses the scoring method of the estimator. For a classifier,
# it's the mean accuracy
rnd_search = RandomizedSearchCV(tree_clf, param_distributions=param_distrib,
                                n_iter=5000, cv=5,
                                scoring='accuracy',
                                random_state=42)
if do_rnd_search:
    logger.info("Doing RandomizedSearchCV with:")
    logger.info(param_distrib)
    rnd_search.fit(X_train, y_train)
    # Save search
    pkl_path = Path(__file__).parents[0] / "pkls"
    if not pkl_path.is_dir():
        pkl_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(rnd_search, pkl_path / "rnd_search_{}.pkl".format(mod_name))
else:
    pkl_path = Path(__file__).parents[0] / "pkls"
    logger.info("Loading RandomizedSearchCV:")
    rnd_search = joblib.load(pkl_path / "rnd_search_{}.pkl".format(mod_name))
    
logger.info(rnd_search.best_params_)
logger.info(rnd_search.best_estimator_)
cv_res = fn.get_cv_results(rnd_search)
logger.info(cv_res)
# fn.display_feature_importances(rnd_search, df_housing_prepared.columns)


########## Test best model ##########
# Define best model
final_model = rnd_search.best_estimator_
logger.info("Final model is:\n{}".format(final_model))

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
