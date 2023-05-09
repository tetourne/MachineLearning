from MachineLearning import functions as fn
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score


random_forest = False
do_grid_search = True
do_plot = False

########## Load data ##########
mnist = fn.load_mnist()
X = mnist.data
y = mnist.target
# split train and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
# shuffle the training set to have similiar cross-validation folds
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]


########## Binary Classifier ##########
# recognize digit '5'
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
print("Fitting SGD Classifier...")
sgd_clf.fit(X_train, y_train_5)


########## Cross Validation ##########
### Accuracy:
# how much good predictions have been done
accuracies = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# accuracies = fn.cross_val_accuracy(sgd_clf, X_train, y_train_5, cv=3)  # recoding of sklearn's function
print("Accucaries: {}".format(accuracies))

### Confusion Matrix:
# [[TN, FP],[FN, TP]]
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print("Confusion matrix:")
print(cm)

### Precision and recall
# preci: TP/(TP+FP) : ratio of good positive predictions
# recall or sensitivity of true positive rate: TP/(TP+FN) : ratio of positive instances corretly detected by the classifier
preci = precision_score(y_train_5, y_train_pred)
print("Precision: {}".format(preci))
recall = recall_score(y_train_5, y_train_pred)
print("Recall: {}".format(recall))
f1score = f1_score(y_train_5, y_train_pred)
print("F1 score: {}".format(f1score))

### Precision / recall trade-off:
threshold = 3000
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
if do_plot:
    fn.plot_preci_recall_tradeoff(precisions, recalls, thresholds, threshold=threshold)
    fn.plot_preci_vs_recall(precisions, recalls, thresholds, threshold=threshold)

### Find threshold for a given precision
precision_mini = 90
idx_for_precision_mini = (precisions >= precision_mini).argmax()
threshold_for_precision_mini = thresholds[idx_for_precision_mini]
print("Threshold to acheive {} precision: {}".format(precision_mini, threshold_for_precision_mini))
y_train_pred_preci_mini = (y_scores >= threshold_for_precision_mini)
print("Precision for this threshold:{}".format(precision_score(y_train_5, y_train_pred_preci_mini)))
recall_at_precision_mini = recall_score(y_train_5, y_train_pred_preci_mini)
print("Recall for this threshold:{}".format(recall_at_precision_mini))

### ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_train_5, y_scores)
if do_plot:
    fn.plot_roc_curve(fpr, tpr, thresholds_roc, threshold=threshold)
print("ROC AUC score: {}".format(roc_auc_score(y_train_5, y_scores)))
      
### Compare with Random Forest Classifier
if random_forest:
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")  # for each instance, return a probability for each class. Here it's 2
    y_scores_forest = y_probas_forest[:, 1]  # consider the score as the probability of positive class
    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
    if do_plot:
        fn.plot_preci_vs_recall(precisions, recalls, thresholds, last_curve=False)
        fn.plot_preci_vs_recall(precisions_forest, recalls_forest, thresholds_forest)


########## Trying KNeighbors Classifier ##########
kneighbors_clf = KNeighborsClassifier()

# explore the model hyperparameters with grid search
param_grid = [
    {'n_neighbors': [2, 4, 6, 8], 'weights': ['uniform', 'distance']}
]
grid_search = GridSearchCV(kneighbors_clf, param_grid, cv=5)
if do_grid_search:
    print("Doing GridSearchCV with:")
    print(param_grid)
    grid_search.fit(X_train, y_train)
    # Save search
    pkl_path = Path(__file__).parents[0] / "pkls"
    if not pkl_path.is_dir():
        pkl_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search, pkl_path / "grid_search_kneighbors_clf.pkl")
if not do_grid_search:
    pkl_path = Path(__file__).parents[0] / "pkls"
    print("Loading GridSearchCV:")
    grid_search = joblib.load(pkl_path / "grid_search_kneighbors_clf.pkl")
grid_search.best_params_
grid_search.best_estimator_
fn.display_cv_results(grid_search)
# fn.display_feature_importances(grid_search)


### Evaluate best model
best_model = grid_search.best_estimator_
print("Best model is:")
print(best_model)

# on train set
print("Evaluate best model on train set:")
y_train_pred = cross_val_predict(best_model, X_train, y_train, cv=10)
preci = precision_score(y_train, y_train_pred, average='macro')  # The average='macro' parameter calculates the precision for each class separately and then takes the average, giving equal weight to each class.
print("Precision: {}".format(preci))
recall = recall_score(y_train, y_train_pred, average='macro')
print("Recall: {}".format(recall))
f1score = f1_score(y_train, y_train_pred, average='macro')
print("F1 score: {}".format(f1score))

# on test set
print("Evaluate best model on test set:")
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
preci = precision_score(y_test, y_test_pred, average='macro')
print("Precision: {}".format(preci))
recall = recall_score(y_test, y_test_pred, average='macro')
print("Recall: {}".format(recall))
f1score = f1_score(y_test, y_test_pred, average='macro')
print("F1 score: {}".format(f1score))


########## Trying data augmentation ##########
# Augment data with shifted images
dx, dy = 3, 3
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]
for ddx, ddy in ((-dx, -dy), (-dx, dy), (dx, -dy), (dx, dy)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(fn.shift_image(image, ddx, ddy))
        y_train_augmented.append(label)
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Shuffle the augmented training set
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

# Evaluate best model with augmented training set
# on train set
print("Evaluate best model on augmented train set:")
y_train_augmented_pred = cross_val_predict(best_model, X_train_augmented, y_train_augmented, cv=10)
preci = precision_score(y_train_augmented, y_train_augmented_pred, average='macro')  # The average='macro' parameter calculates the precision for each class separately and then takes the average, giving equal weight to each class.
print("Precision: {}".format(preci))
recall = recall_score(y_train_augmented, y_train_augmented_pred, average='macro')
print("Recall: {}".format(recall))
f1score = f1_score(y_train_augmented, y_train_augmented_pred, average='macro')
print("F1 score: {}".format(f1score))

# on test set
print("Evaluate best model on test set with augmented train set:")
best_model.fit(X_train_augmented, y_train_augmented)
y_test_pred = best_model.predict(X_test)
preci = precision_score(y_test, y_test_pred, average='macro')
print("Precision: {}".format(preci))
recall = recall_score(y_test, y_test_pred, average='macro')
print("Recall: {}".format(recall))
f1score = f1_score(y_test, y_test_pred, average='macro')
print("F1 score: {}".format(f1score))
