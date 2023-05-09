from MachineLearning import functions as fn
from MachineLearning import transformers as tr
import pandas as pd
import numpy as np
import scipy as sp
import joblib
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error


########## SETUP ##########
visualize_data = False
train_model = False
do_grid_search = False
do_rnd_search = False


########## Read and split data ##########
filename = "/home/thomas/Documents/Python/Tutorials/Hands_on_ML/handson-ml/datasets/housing/housing.csv"
housing, test_set = fn.read_and_split_data(filename)

########## Visualize data ##########
if visualize_data:
    # scatter matrix
    attributes = housing.columns
    # attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
    fn.save_fig("scatter_matrix_plot")  # extra code
    plt.show()
    
    # correlations
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    
    # Attribute combinations
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


########## Clean Data ##########
# Get instances and labels
housing_labels = housing["median_house_value"].copy()
housing.drop("median_house_value", axis=1, inplace=True)

# Split numerical and categorical data
housing_num = housing.select_dtypes(include=[np.number])
housing_cat = housing.select_dtypes(include=[object])
num_attribs = list(housing_num)
cat_attribs = list(housing_cat)

# # num transformers
# selector_num = tr.DataFrameSelector(num_attribs)
# imputer = SimpleImputer(strategy="median")
# attribs_adder = tr.CombinedAttributesAdder()
# std_scaler = StandardScaler()
# # cat transformers
# selector_cat = tr.DataFrameSelector(cat_attribs)
# one_hot_encoder = OneHotEncoder(sparse_output=False)

# # Apply num transformers manually
# h1_num = selector_num.fit_transform(housing)
# h2_num = imputer.fit_transform(h1_num)
# h3_num = attribs_adder.fit_transform(h2_num)
# h4_num = std_scaler.fit_transform(h3_num)
# # Apply cat transformers
# h1_cat = selector_cat.fit_transform(housing)
# h2_cat = one_hot_encoder.fit_transform(h1_cat)
# h3_cat = pd.DataFrame(h2_cat, columns=one_hot_encoder.get_feature_names_out())

### There is an example in the jupyter notebook that shows how to create a transformer
### which identifies the clusters in (Latitude, Longitude) (districts) where the
### median house value is the highest (using sklearn.cluster.KMeans)

# Create pipeline with all transformers
# num_pipeline = Pipeline([
#     # ('selector', tr.DataFrameSelector(num_attribs)), ### replaced by ColumnTransformer
#     ('imputer', SimpleImputer(strategy="median")),
#     ('attribs_adder', tr.CombinedAttributesAdder()),
#     ('std_scaler', StandardScaler()),
# ])
# num_pipeline = make_pipeline(  ### make_pipeline is an alternative to Pipeline
#     tr.DataFrameSelector(num_attribs),
#     SimpleImputer(strategy="median"),
#     tr.CombinedAttributesAdder()
# #     StandardScaler()
# )
# cat_pipeline = Pipeline([
#     # ('selector', tr.DataFrameSelector(cat_attribs)),  ### replaced by ColumnTransformer
#     ('imputer', SimpleImputer(strategy="most_frequent")),
#     ('onehotencoder', OneHotEncoder(sparse_output=False)),
# ])
# cat_pipeline = make_pipeline( ### alternative to Pipeline
#     tr.DataFrameSelector(cat_attribs),
# #     LabelBinarizer()
# )

# preprocessing = ColumnTransformer([  ### Alternative to FeatureUnion
#     ("num", num_pipeline, num_attribs),
#     ("cat", cat_pipeline, cat_attribs),
# ])
# preprocessing = make_column_transformer(  ### Alternative to ColumnTransformer
#     (num_pipeline, make_column_selector(dtype_include=np.number)),
#     (cat_pipeline, make_column_selector(dtype_include=object)),
# )
# housing_prepared = preprocessing.fit_transform(housing)

# preprocessing = FeatureUnion(transformer_list=[  ### Combine num and cat pipeline ; need DataFrameSelector in each pipeline
#     ("num_pipeline", num_pipeline),
#     ("cat_pipeline", cat_pipeline),
# ])
# housing_prepared = preprocessing.fit_transform(housing)

### A more complexe pipeline that allows to use pipeline.get_feature_names_out()
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


########## Select and train models ##########
### The fitting of model can be include into a pipeline that does the transformation
### of data and then the fitting of the model :
### lin_reg = make_pipeline(preprocessing, LinearRegression())
### lin_reg.fit(housing, housing_labels)

# Linear regressor
lin_reg = LinearRegression()
if train_model:
    print("Training Linear Regressor")
    fn.eval_model(lin_reg, housing_prepared, housing_labels)

# Decision Tree regressor
tree_reg = DecisionTreeRegressor()
if train_model:
    print("Training Decision Tree Regressor")
    fn.eval_model(tree_reg, housing_prepared, housing_labels)

# Random Forest regressor
forest_reg = RandomForestRegressor()
if train_model:
    print("Training Random Forest Regressor")
    fn.eval_model(forest_reg, housing_prepared, housing_labels)

### Use GridSearchCV to fine tune the models
param_grid = [
    {'n_estimators': [10, 30, 100], 'max_features': [6, 8, 10, 12]},
    {'bootstrap': [False], 'n_estimators': [10, 30], 'max_features': [4, 8, 12]},
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_root_mean_squared_error')
if do_grid_search:
    print("Doing GridSearchCV with:")
    print(param_grid)
    grid_search.fit(housing_prepared, housing_labels)
    grid_search.best_params_
    grid_search.best_estimator_
    fn.display_cv_results(grid_search)
    fn.display_feature_importances(grid_search, df_housing_prepared.columns)

### Randomized Search
# better to use randomized search when the hyperparameter space is large

param_distrib = {'n_estimators': sp.stats.randint(low=2, high=120),
                 'max_features': sp.stats.randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distrib,
                                n_iter=20, cv=5,
                                scoring='neg_root_mean_squared_error', random_state=42)
if do_rnd_search:
    print("Doing RandomizedSearchCV with:")
    print(param_distrib)
    rnd_search.fit(housing_prepared, housing_labels)
    rnd_search.best_params_
    rnd_search.best_estimator_
    fn.display_cv_results(rnd_search)
    fn.display_feature_importances(rnd_search, df_housing_prepared.columns)

### These are other random generators that can be use to define the param_distributions
# scipy.stats.randint(a, b+1): for hyperparameters with discrete values that range from a to b, and all values in that range seem equally likely.
# scipy.stats.uniform(a, b): this is very similar, but for continuous hyperparameters.
# scipy.stats.geom(1 / scale): for discrete values, when you want to sample roughly in a given scale. E.g., with scale=1000 most samples will be in this ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
# scipy.stats.expon(scale): this is the continuous equivalent of geom. Just set scale to the most likely value.
# scipy.stats.reciprocal(a, b): when you have almost no idea what the optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then you're just as likely to sample a value between 0.01 and 0.1 as a value between 10 and 100.

### Select and save best model
# final_model = rnd_search.best_estimator_
final_model = RandomForestRegressor(bootstrap=False, max_features=6, n_estimators=200)  # got from rnd search + grid search
final_model.fit(housing_prepared, housing_labels)  # only needed if the model is redefined the line above
print("Evaluating final model on training set...")
fn.eval_model(final_model, housing_prepared, housing_labels)  # last eval before comparing with test set

# build full pipeline and save
final_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", final_model),
    ])
model_path = Path(__file__).parents[0] / "models"
if not model_path.is_dir():
    model_path.mkdir(parents=True, exist_ok=True)

joblib.dump(final_pipeline, model_path / "final_pipeline.pkl")

########## Evaluate the model on the test set ##########
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

# # Transform test set
# X_test_prepared = preprocessing.transform(X_test)
# df_X_test_prepared = pd.DataFrame(  # Use DataFrame if necessary
#     X_test_prepared,
#     columns=preprocessing.get_feature_names_out(),
#     index=X_test.index)  # looks like the indices are shuffled if index is not specified
# # ... and make Prediction
# final_predictions = final_model.predict(X_test_prepared)

# Or directly make prediction
final_predictions = final_pipeline.predict(X_test)

# Evaluate model
print("Evaluating final model on test set...")
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print("Final RMSE on test set is: {}".format(final_rmse))
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print("{}% confidence interval on RMSE is:".format(confidence*100))
print(np.sqrt(sp.stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=sp.stats.sem(squared_errors))))


########## Reload the model and make prediction on new data ##########
final_pipeline_reloaded = joblib.load(model_path / "final_pipeline.pkl")
new_data = housing.iloc[:5]  # pretend these are new districts
predictions = final_pipeline_reloaded.predict(new_data)
