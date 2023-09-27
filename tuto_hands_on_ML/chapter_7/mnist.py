import sys
import logging
import joblib
import numpy as np
import scipy as sp
import pandas as pd
from pathlib import Path
from scipy.stats import loguniform, uniform, randint
from MachineLearning import functions as fn
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier


do_rnd_search_rnd_clf = False
do_rnd_search_extra_clf = False
do_rnd_search_svm_clf = False
train_voting_clf = True

########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / sys.argv[0].replace('.py', '.log')
logger = fn.my_logger(logs_file)


########## Load data ##########
logger.info("Loading MNIST dataset")
mnist = fn.load_mnist()
X_mnist = mnist.data
y_mnist = mnist.target

# split train and test sets
train_length = 60000  # 60000 for the full MNIST dataset
X_train, X_test = X_mnist[:train_length], X_mnist[train_length:]
y_train, y_test = y_mnist[:train_length], y_mnist[train_length:]

# shuffle the training set to have similiar cross-validation folds
np.random.seed(42)
shuffle_index = np.random.permutation(train_length)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# Split train set into train and validation sets
train_length = 50000
X_train, X_val = X_train[:train_length], X_train[train_length:]
y_train, y_val = y_train[:train_length], y_train[train_length:]


########## Train Random Forest classifier ##########
rnd_clf = RandomForestClassifier(n_jobs=-1)
mod_name = 'rnd_clf2'
subset_train_length = 10000
param_distrib = {
    'n_estimators': randint(50,500),
    'max_depth': randint(10,100),
    # 'min_samples_split': randint(2, 10),
    # 'min_samples_leaf': randint(1, 10)
    # 'max_leaf_nodes': randint(10,100)
}
# scoring methods can be accuracy, precision, recall, f1score, roc_auc...
# By default, it uses the scoring method of the estimator. For a classifier,
# it's the mean accuracy
rnd_search = RandomizedSearchCV(rnd_clf, param_distributions=param_distrib,
                                n_iter=10, cv=3,
                                # scoring='roc_auc',
                                random_state=42, verbose=3)

# nothing to modify bellow
if do_rnd_search_rnd_clf:
    logger.info(f"Doing RandomizedSearchCV for model {mod_name}\n" +
                "with {subset_train_lemgth} instances \n" +
                "and parameter distribution:")
    logger.info(param_distrib)
    rnd_search.fit(X_train[:subset_train_length], y_train[:subset_train_length])
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

# best_rnd_clf = rnd_search.best_estimator_
best_rnd_clf = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1)



########## Train Extra-Trees classifier ##########
extra_clf = ExtraTreesClassifier(n_estimators=200, max_depth=30, n_jobs=-1)
mod_name = 'extra_clf'
subset_train_length = 10000
param_distrib = {
    # 'n_estimators': randint(100,500),
    # 'max_depth': randint(10,40),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    # 'max_leaf_nodes': randint(10,100)
}
# scoring methods can be accuracy, precision, recall, f1score, roc_auc...
# By default, it uses the scoring method of the estimator. For a classifier,
# it's the mean accuracy
rnd_search = RandomizedSearchCV(extra_clf, param_distributions=param_distrib,
                                n_iter=20, cv=3,
                                # scoring='roc_auc',
                                random_state=42, verbose=3)

# nothing to modify bellow
if do_rnd_search_extra_clf:
    logger.info(f"Doing RandomizedSearchCV for model {mod_name}\n" +
                "with {subset_train_lemgth} instances \n" +
                "and parameter distribution:")
    logger.info(param_distrib)
    rnd_search.fit(X_train[:subset_train_length], y_train[:subset_train_length])
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

# best_extra_clf = rnd_search.best_estimator_
best_extra_clf = ExtraTreesClassifier(n_estimators=200, max_depth=30, n_jobs=-1)


########## Train SVM classifier ##########
svm_clf = Pipeline([
    ('standardscaler', StandardScaler()),  # necessary for SVM classifier
    ('svc', SVC())
])
mod_name = "svm_clf"
subset_train_length = 2000
param_distrib = {
    'svc__kernel': ['poly'],
    'svc__gamma': loguniform(1e-2, 1),
    'svc__C': loguniform(0.1, 10)
}
# scoring methods can be accuracy, precision, recall, f1score, roc_auc...
# By default, it uses the scoring method of the estimator. For a classifier,
# it's the mean accuracy
rnd_search = RandomizedSearchCV(svm_clf, param_distributions=param_distrib,
                                n_iter=30, cv=3,
                                # scoring='roc_auc',
                                random_state=42, verbose=3)

# nothing to modify bellow
if do_rnd_search_svm_clf:
    logger.info(f"Doing RandomizedSearchCV for model {mod_name}\n" +
                "with {subset_train_lemgth} instances \n" +
                "and parameter distribution:")
    logger.info(param_distrib)
    rnd_search.fit(X_train[:subset_train_length], y_train[:subset_train_length])
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

# best_svm_clf = rnd_search.best_estimator_
best_svm_clf = Pipeline([
    ('standardscaler', StandardScaler()),  # necessary for SVM classifier
    ('svc', SVC(kernel='poly', gamma=1, C=1))
])



########## Train Voting classifier ##########
### Set probability = True, to be able to use voting='soft'
best_svm_clf.set_params(svc__probability=True)

### Define and train Voting classifier
# estimators=[('rnd', best_rnd_clf), ('extra', best_extra_clf), ('svm', best_svm_clf)]
named_estimators = [("random_forest_clf", best_rnd_clf),
              ("extra_trees_clf", best_extra_clf),
              ("svm_clf", best_svm_clf),
              ]
voting_clf = VotingClassifier(
    estimators=named_estimators,
    voting='soft',
)
if train_voting_clf:
    subset_train_length = 50000
    logger.info(f"Training a Voting classifier with\n{voting_clf}")
    voting_clf.fit(X_train[:subset_train_length], y_train[:subset_train_length])

### Test and compare Voting classifier to individual classifiers
# clfs = [best_rnd_clf, best_extra_clf, best_svm_clf, voting_clf]
# voting_clf does a clone of all estimators and trains the clone using
# class indices as the labels. So it's necessary to encode the labels.
clfs = voting_clf.estimators_
# clfs.append(voting_clf)
y_val_encoded = y_val.astype(np.int64)
y_val_preds = []
accuracies = []
precisions = []
recalls = []
f1scores = []

if train_voting_clf:
    for clf in clfs+[voting_clf]:
        logger.info(f"Doing prediction on the validation set for: {clf.__class__.__name__}")
        y_val_pred = clf.predict(X_val).astype(np.int64)
        y_val_preds.append(y_val_pred)
        logger.info("Measuring accuracy, precision, recall and f1 score")
        accu = accuracy_score(y_val_encoded, y_val_pred)
        accuracies.append(accu)
        logger.info(f"Accuracy on validation set is {accu}")
        preci = precision_score(y_val_encoded, y_val_pred, average='macro')
        precisions.append(preci)
        logger.info(f"Precision on validation set is {preci}")
        recall = recall_score(y_val_encoded, y_val_pred, average='macro')
        recalls.append(recall)
        logger.info(f"Recall on validation set is {recall}")
        f1score = f1_score(y_val_encoded, y_val_pred, average='macro')
        f1scores.append(f1score)
        logger.info(f"F1 score on validation set is {f1score}")

results = pd.DataFrame([accuracies, precisions, recalls, f1scores],
                       columns=['RandomeForest', 'ExtraTree', 'SVM', 'Voting'],
                       index=['Accuracy', 'Precision', 'Recall', 'F1 score']
                       )
logger.info(f"Summary of results:\n{results}")

########## Stacking Ensemble ##########
### Create new training set from prediction of each classifier
# don't include predictions from voting classifier
X_val_pred = (np.array(y_val_preds)[:-1, :]).T  # (n_instance, n_estimators)

### Define blender
# oob_score option allows to keep unseen instances (for each classifier)
# to measure accuracy without using a validation set.
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True,
                                            random_state=42, n_jobs=-1)
logger.info(f"Training blender: {rnd_forest_blender}")
rnd_forest_blender.fit(X_val_pred, y_val)
logger.info("Accuracy for blender is:")
accuracy = rnd_forest_blender.oob_score_
logger.info(accuracy)


########## Performances on test set ##########
# estimators = [best_rnd_clf, best_extra_clf, best_svm_clf]
estimators = voting_clf.estimators_
logger.info("\n")
logger.info("Perfomances on the test set")
X_test_pred = np.empty((len(X_test), len(estimators)), dtype=object)
logger.info(f"Doing prediction on the test set for:\n {estimators}")
for i, estimator in enumerate(estimators):
    X_test_pred[:, i] = estimator.predict(X_test)
y_pred = rnd_forest_blender.predict(X_test_pred)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Accuracy on the test set is: {accuracy}")


########## Trying StackingClassifier ##########
# Since StackingClassifier uses K-Fold cross-validation, we don't need
# a separate validation set, so let's join the training set and the
# validation set into a bigger training set.
X_train_full, y_train_full = X_mnist[:60000], y_mnist[:60000]
stack_clf = StackingClassifier(named_estimators,
                               final_estimator=rnd_forest_blender)
# Takes a long time as it uses K-fold validation
logger.info("Training a Stacking Classifier")
stack_clf.fit(X_train_full, y_train_full)
accuracy = stack_clf.score(X_test, y_test)
logger.info(f"Accuracy is {accuracy}")
