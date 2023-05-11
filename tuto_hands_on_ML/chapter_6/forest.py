import sys
import logging
import joblib
import numpy as np
import scipy as sp
from pathlib import Path
from scipy.stats import loguniform, uniform, randint, mode
from matplotlib import pyplot as plt
from MachineLearning import functions as fn
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score, cross_val_predict, ShuffleSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


plot_dataset = False
do_cross_val = False
do_rnd_search = True


########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / 'forest.log'
logger = fn.my_logger(logs_file)

########## Load dataset ##########
# generate dataset
logger.info("Generating moons data")
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

# Generate n_trees subsets of n_instances each. They are drawn from X_train.
n_trees = 1000
n_instances = 100
logger.info(f"Generating {n_trees} subsets containing {n_instances} instances each")

mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances,
                  random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

########## Fit models on each subset ##########
# best_model = DecisionTreeClassifier(criterion='entropy', max_depth=22, max_leaf_nodes=32,
#                        min_samples_leaf=58, min_samples_split=46,
#                        random_state=42)
best_model = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=17, random_state=42)
logger.info(f"Fitting best model:\n{best_model} \n on each subsets")
forest = [clone(best_model) for _ in range(n_trees)]
accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

logger.info(f"Averaged accuracy is: {np.mean(accuracy_scores)}")

########## Select the most frequent prediction ##########
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
accuracy = accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
logger.info(f"Accuracy from the forest: {accuracy}")
