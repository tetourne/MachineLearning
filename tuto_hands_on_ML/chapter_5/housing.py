from MachineLearning import functions as fn
from MachineLearning import transformers as tf
import numpy as np
import scipy as sp
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import loguniform, uniform
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


do_rnd_search = False


########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / 'housing.log'
logger = fn.my_logger(logs_file)


########## Read and split data ##########
filename = "/home/thomas/Documents/Python/Tutorials/Hands_on_ML/handson-ml/datasets/housing/housing.csv"
logger.info("Reading data from {}".format(filename))
housing, test_set = fn.read_and_split_data(filename)


########## Clean Data ##########
# Get instances and labels
logger.info("Cleaning dataset...")
housing_labels = housing["median_house_value"].copy()
housing.drop("median_house_value", axis=1, inplace=True)

# Split numerical and categorical data
housing_num = housing.select_dtypes(include=[np.number])
housing_cat = housing.select_dtypes(include=[object])
num_attribs = list(housing_num)
cat_attribs = list(housing_cat)

# Build pipeline and prepare data
log_pipeline = make_pipeline(  # pipeline which takes the log of a num series: it's correcting for the skewed distributions (making them more symetric)
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
default_num_pipeline = make_pipeline(  # use as the default pipeline in ColumnTransformer()
    SimpleImputer(strategy="median"), 
    StandardScaler())
cat_pipeline = make_pipeline(  # pipeline for categorical series
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(sparse_output=False))
preprocessing = ColumnTransformer([
        ("bedrooms", fn.ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", fn.ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", fn.ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        # ("geo", cluster_simil, ["latitude", "longitude"]), not implemented, the code is in the notebook
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age. If no transformer specified, the default is to drop the columns
housing_prepared = preprocessing.fit_transform(housing)
df_housing_prepared = pd.DataFrame(
    housing_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=housing.index)  # looks like the indices are shuffled if index is not specified


########## Tune model ##########
### Define model
# svm_reg = LinearSVR(fit_intercept=True)
svm_reg = SVR()
mod_name = "svm_reg_1"

### Randomized Search
param_distrib = {
    'kernel': ['rbf'],
    'gamma': uniform(0, 0.4),
    'C': uniform(1e5, 4e5)
}
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distrib,
                                n_iter=100, cv=3,
                                scoring='neg_root_mean_squared_error', random_state=42,
                                verbose=3)
if do_rnd_search:
    logger.info("Doing RandomizedSearchCV with:")
    logger.info(param_distrib)
    rnd_search.fit(housing_prepared[:2000], housing_labels[:2000])
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
final_model = SVR(kernel='rbf', C=225742, gamma=0.203428)
logger.info("Testing best model:")
logger.info(final_model)
fn.eval_model(final_model, housing_prepared, housing_labels, cv=5)

