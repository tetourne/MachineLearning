from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml
from zlib import crc32
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import scipy as sp
import tarfile
import urllib.request
import joblib
import os
import sys
import logging
from typing import Tuple, Union


########## Logger ##########
def my_logger(filename, logger_level=logging.DEBUG, console_level=logging.DEBUG, logfile_level=logging.INFO, logs_format='%(asctime)s - %(levelname)s - %(message)s'):
    """
    Create a logger object with a file handler and a console handler.

    :param filename: The name of the log file.
    :type filename: str
    :param logger_level: The level of the logger. Default is logging.DEBUG.
    :type logger_level: int
    :param console_level: The level of the console handler. Default is logging.DEBUG.
    :type console_level: int
    :param logfile_level: The level of the file handler. Default is logging.INFO.
    :type logfile_level: int
    :param logs_format: The format of the log messages. Default is '%(asctime)s - %(levelname)s - %(message)s'.
    :type logs_format: str
    :return: A logger object with a file handler and a console handler.
    :rtype: logging.Logger
    """
    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logger_level)

    # Create a file handler and set the level to DEBUG
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logfile_level)

    # Create a console handler and set the level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # Create a formatter and add it to the handlers
    class MultiLineFormatter(logging.Formatter):
        def format(self, record):
            # Add a newline character before the message if the message contains multiple lines
            if '\n' in record.getMessage():
                record.msg = '\n' + record.getMessage()
            return super().format(record)

    # Create a formatter and add it to the handlers
    formatter = MultiLineFormatter(logs_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    if len(logger.handlers) < 1:
        logger.addHandler(file_handler)
    if len(logger.handlers) < 2:
        logger.addHandler(console_handler)

    return logger

# Import the logger from the main module
logger = logging.getLogger(__name__)


########## Reading data ##########
def load_housing_data():
    """
    Load the California housing dataset from a local file or from a remote URL.

    :return: A pandas DataFrame containing the California housing dataset.
    :rtype: pd.DataFrame

    :raises urllib.error.URLError: If the download of the dataset fails.

    """
    tarball_path = Path("datasets/housing.csv")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/handson-ml/tree/master/datasets/housing/housing.csv"
        urllib.request.urlretrieve(url, tarball_path)
        # with tarfile.open(tarball_path) as housing_tarball:
        #     housing_tarball.extractall(path="Datasets")
    return pd.read_csv(Path("datasets/housing.csv"))


def load_mnist(name='mnist_784', path='../datasets', as_frame=False, force_reload=False):
    """
    Load the MNIST dataset from a local file or from the OpenML repository.

    :param name: The name of the MNIST dataset to load. Defaults to 'mnist_784'.
    :type name: str
    :param path: The path where to cache the dataset. Defaults to 'Datasets'.
    :type path: str
    :param as_frame: Whether to return the data as a pandas DataFrame. Defaults to False.
    :type as_frame: bool
    :param force_reload: Whether to force a reload of the dataset from the OpenML repository. Defaults to False.
    :type force_reload: bool
    :return: A dictionary or a memory cache object containing the MNIST dataset.
    :rtype: Union[joblib.Memory, dict]

    :raises ValueError: If an invalid value is provided for the `as_frame` parameter.

    """
    path = Path() / path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    filename = name + '.pkl'
    full_path = path / filename
    if force_reload:
        mnist = fetch_openml(name, as_frame=as_frame)
        joblib.dump(mnist, full_path)
    elif os.path.isfile(full_path):
        mnist = joblib.load(full_path)
    else:
        mnist = fetch_openml(name, as_frame=as_frame)
        joblib.dump(mnist, full_path)
    return mnist
    


########## Splitting data ##########
def read_and_split_data(filename, random_state=42):
    """
    Read dataset from a CSV file and split into train and test sets using stratified suffled split.

    :param filename: The name of the CSV file to load.
    :type name: str
    :param random_state: The seed of the random generator used to shuffle the data. Defaults to 42.
    :type random_state: int
    :return: The train and test datasets.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # Read data
    housing = pd.read_csv(filename)
    # create income_cat attribute to linmite the number of income categories
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    # do stratified suffle split using sklearn, based on income_cat
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=random_state)
    # remove income_cat attribute
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    # use only train set
    return strat_train_set.copy(), strat_test_set.copy()


def shuffle_and_split_data(data, test_ratio, random_state=42):
    """
    Shuffle data and split it into a train and a test sets.

    :param data: The dataset to shuffle and split.
    :type data: Union[np.ndarray, pd.DataFrame]
    :param test_ratio: The ratio of test data over all data. Range from 0 to 1. Usually chosen around 0.2.
    :type test_ratio: float
    :param random_state: the seed of the random generator used to shuffle the data. Defaults to 42.
    :type random_state: int
    :return: The train and test datasets.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def split_data_with_id_hash(data, test_ratio, id_column):
    """
    Split dataset using the IDs from a column of the DataFrame.

    :param data: The dataset to splot
    :type data: pd.DataFrame
    :param test_ratio: The ratio of test data over all data. Range from 0 to 1. Usually chosen around 0.2.
    :type test_ratio: float
    :param id_column: The name of the column to use as IDs.
    :type id_column: str
    :return: The train and test datasets.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]



########## Preprocessing ##########
def shift_image(image, dx, dy, size=(28,28)):
    """
    Shifts an image by a specified number of pixels in the x and y directions.

    :param image: The image to shift.
    :type image: numpy.ndarray
    :param dx: The number of pixels to shift the image in the x direction.
    :type dx: float
    :param dy: The number of pixels to shift the image in the y direction.
    :type dy: float
    :param size: The dimensions of the image to be reshaped before shifting. Defaults to (28, 28).
    :type size: tuple
    :return: The shifted image.
    :rtype: numpy.ndarray
    """
    image = image.reshape(size)
    shifted_image = sp.ndimage.shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


def column_ratio(X):
    """
    Return the ratio of the 2 series of a DataFrame.

    :param X: The dataframe containing the two series to calculate the ratio of.
    :type X: pd.DataFrame
    :return: The ratio of the two series in the dataframe.
    :rtype: np.ndarray
    """
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    """
    Function needed in ratio_pipeline() to be able to use get_feature_names_out().

    :param function_transformer: A transformer that implements a function.
    :type function_transformer: sklearn.preprocessing.FunctionTransformer
    :param feature_names_in: The input features.
    :type feature_names_in: list
    :return: A list of feature names out.
    :rtype: list
    """
    return ["ratio"]


def ratio_pipeline():
    """
    One pipeline that does the ratio of 2 dataframe series.

    :return: The pipeline for calculating the ratio of 2 dataframe series.
    :rtype: sklearn.pipeline.Pipeline
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


########## Model Selection ##########
def eval_model(model, X, y, cv=10):
    """
    This function prints the training error and the validation errors. If the training error is much smaller than the validation errors, it means that the model is overfitting the data.

    :param model: The model to evaluate.
    :type model: sklearn.base.BaseEstimator
    :param X: The feature matrix.
    :type X: Union[numpy.ndarray, pandas.DataFrame]
    :param y: The target vector (labels).
    :type y: Union[numpy.ndarray, pandas.Series]
    :param cv: The number of cross-validation subsets (folds). Defaults to 10.
    :type cv: int
    """
    model.fit(X, y)
    prediction = model.predict(X)
    rmse = mean_squared_error(y, prediction, squared=False)
    logger.info("Training error is {}".format(rmse))
    # Validation error
    rmses = -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    logger.info("Validation errors :")
    logger.info(pd.Series(rmses).describe())


def get_cv_results(search):
    """
    This function returns the results of CV search as a nice pandas.DataFrame.

    :param search: The GridSearchCV object.
    :type search: sklearn.model_selection.GridSearchCV
    :return: The sorted dataframe containing CV results
    :rtype: pd.DataFrame
    """
    cv_res = pd.DataFrame(search.cv_results_)
    param_list = [item for item in cv_res.columns if "param_" in item]
    # score_list = [item for item in cv_res.columns if "_test_score" in item]
    score_list = ["mean_test_score", "std_test_score"]
    cv_res = cv_res[param_list + score_list]
    param_list = [item[6:] for item in param_list]
    score_list = [item[:-6] for item in score_list]
    cv_res.columns = param_list + score_list
    cv_res[score_list] = -cv_res[score_list]
    # logger.info("\nResults of CV search:")
    # logger.info(cv_res)
    # logger.info("\nSorted results of CV search:")
    return cv_res.sort_values("mean_test")


def display_feature_importances(search, df_columns=None):
    """
    Display the importances of each feature for a CV search exploration.

    :param search: The GridSearchCV object.
    :type search: sklearn.model_selection.GridSearchCV
    :param df_columns: The list of all features. Defaults to None.
    :type df_columns: list
    """
    feature_importances = search.best_estimator_.feature_importances_
    if df_columns is None:
        res = sorted(feature_importances, reverse=True)
    else:
        res = sorted(zip(feature_importances, df_columns), reverse=True)
    for r in res:
        logger.info(r)
    

def cross_val_accuracy(model, X, y, cv=3):
    """
    Compute the cross-validation accuracy for a given model and dataset.

    :param model: Any model from sklearn that implements the scikit-learn estimator interface.
    :type model: sklearn.linear_model.SGDClassifier
    :param X: The feature matrix.
    :type X: Union[numpy.ndarray, pandas.DataFrame]
    :param y: The target vector (labels).
    :type y: Union[numpy.ndarray, pandas.Series]
    :param cv: The number of folds in the cross-validation. Defaults to 3.
    :type cv: int
    :return: The measured accuracy on each fold.
    :rtype: np.ndarray
    """
    # recoding of sklearn.model_selection.cross_val_score(scoring="accuracy")
    accuracy = []
    skfolds = StratifiedKFold(n_splits=cv)  # add shuffle=True if the dataset is not already shuffled
    for train_index, test_index in skfolds.split(X, y):
        clone_model = clone(model)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]
        clone_model.fit(X_train_folds, y_train_folds)
        y_pred = clone_model.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        accuracy.append(n_correct / len(y_pred))
    return np.array(accuracy)


########## Plotting ##########
def save_fig(fig_id, images_path="images", tight_layout=True, fig_extension="png", resolution=300):
    """
    Save the pyplot figure

    :param fig_id: The name of the figure.
    :type fig_id: str
    :param images_path: The path to the directory in which the figure is saved. Defaults to "Images".
    :type images_path: str
    :param tight_layout: Enable tight_layout for the figure. Defaults to True.
    :type tight_layout: bool
    :param fig_extension: The extension of the file that stores the figure. Defaults to "png".
    :type fig_extension: str
    :param resolution: The resolution of the figure. Defaults to 300.
    :type resolution: int
    """
    path = Path() / images_path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    path = path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_digit(image_data, show=True):
    """
    Plot as an image the digit from MNIST dataset.

    :param image_data: The digit data.
    :type image_data: np.ndarray
    :param show: If true, display the figure. Defaults to True.
    :type show: bool
    """
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    if show:
        plt.show()


def plot_preci_recall_tradeoff(precisions, recalls, thresholds, threshold=3000, figname="precision_recall_vs_threshold_plot", last_curve=True):
    """
    Plot the precision and the recall curves on the same figure, and show their values for a given threshold.

    :param precisions: The precisions.
    :type precisions: np.ndarray
    :param recalls: The recalls.
    :type recalls: np.ndarray
    :param thresholds: The thresholds.
    :type thresholds: np.ndarray
    :param threshold: The threshold to display on the figure. Defaults to 3000.
    :type threshold: int
    :param figname: The name of the figure to save. Defaults to "precision_recall_vs_threshold_plot".
    :type figname: str
    :param last_curve: If False, only plot the precision and recall curves. In order to add more curves to the plot. Defaults to True.
    :type last_curve: bool
    """
    # plt.figure(figsize=(8, 4))  # extra code – it's not needed, just formatting
    plt.plot(thresholds, precisions[:-1], "--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold={}".format(threshold))
    # extra code – this section just beautifies and saves Figure 3–5
    idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
    plt.plot(thresholds[idx], precisions[idx], "ko")
    plt.plot(thresholds[idx], recalls[idx], "ko")
    if last_curve:
        plt.axis([-50000, 50000, 0, 1])
        plt.grid()
        plt.xlabel("Threshold")
        plt.legend(loc="center right")
        save_fig(figname)
        plt.show()


def plot_preci_vs_recall(precisions, recalls, thresholds, threshold=3000, figname="precision_vs_recall_plot", last_curve=True):
    """
    Plot the precision as a function of the recall.

    :param precisions: The precisions.
    :type precisions: np.ndarray
    :param recalls: The recalls.
    :type recalls: np.ndarray
    :param thresholds: The thresholds.
    :type thresholds: np.ndarray
    :param threshold: The threshold to display on the figure. Defaults to 3000.
    :type threshold: int
    :param figname: The name of the figure to save. Defaults to "precision_recall_vs_threshold_plot".
    :type figname: str
    :param last_curve: If False, only plot the precision and recall curves. In order to add more curves to the plot. Defaults to True.
    :type last_curve: bool    
    """
    idx = (thresholds >= threshold).argmax()
    # plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    # extra code – just beautifies and saves Figure 3–6
    plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
    plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
    plt.plot([recalls[idx]], [precisions[idx]], "ko",
             label="Point at threshold {}".format(threshold))
    if last_curve:
        plt.gca().add_patch(patches.FancyArrowPatch(
            (0.79, 0.60), (0.61, 0.78),
            connectionstyle="arc3,rad=.2",
            arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
            color="#444444"))
        plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.axis([0, 1, 0, 1])
        plt.grid()
        plt.legend(loc="lower left")
        save_fig(figname)
        plt.show()


def plot_roc_curve(fpr, tpr, thresholds, threshold=3000, last_curve=True):
    """
    Plot the ROC curve

    :param fpr: The false positive rates.
    :type fpr: np.ndarray
    :param tpr: The truth positive rates.
    :type tpr: np.ndarray
    :param thresholds: The thresholds
    :type thresholds: np.ndarray
    :param threshold: The particular threshold to display. Defaults to 3000.
    :type threshold: int
    :param last_curve: If False, only plot the ROC curve. In order to add more curves to the plot. Defaults to True.
    :type last_curve: bool    

    """
    idx_for_threshold = (thresholds <= threshold).argmax()
    tpr_threshold, fpr_threshold = tpr[idx_for_threshold], fpr[idx_for_threshold]
    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
    plt.plot([fpr_threshold], [tpr_threshold], "ko", label="Threshold={}".format(threshold))
    # extra code – just beautifies and saves Figure 3–7
    if last_curve:
        plt.gca().add_patch(patches.FancyArrowPatch(
            (0.20, 0.89), (0.07, 0.70),
            connectionstyle="arc3,rad=.4",
            arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
            color="#444444"))
        plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
        plt.xlabel('False Positive Rate (Fall-Out)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.grid()
        plt.axis([0, 1, 0, 1])
        plt.legend(loc="lower right", fontsize=13)
        save_fig("roc_curve_plot")
        plt.show()
