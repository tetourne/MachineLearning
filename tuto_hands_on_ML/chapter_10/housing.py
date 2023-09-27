import sys
import numpy as np
import scipy as sp
import pandas as pd
import logging
import joblib
from pathlib import Path

from MachineLearning import functions as fn
import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

do_plots = True
do_training = True


########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / sys.argv[0].replace('.py', '.log')
logger = fn.my_logger(logs_file)


########## Get data ##########
housing = fetch_california_housing()

# Stratified split
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


########## Parameters of DNN ##########
input_shape = X_train.shape[1:]
n_hidden1 = 30
acti_hidden1 = 'relu'
n_hidden2 = 10
acti_hidden2 = 'relu'
n_outputs = 1
acti_outputs = None


########## Define Model ##########
model = keras.models.Sequential([
    keras.layers.Dense(n_hidden1, activation=acti_hidden1, input_shape=input_shape),
    # keras.layers.Dense(n_hidden2, activation=acti_hidden2),
    keras.layers.Dense(n_outputs, activation=acti_outputs)
])


np.random.seed(42)
tf.random.set_seed(42)
keras.backend.clear_session()  # releases the global state of keras

# Defining loss and optimizer
model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              )

# summary of model
logger.info(model.layers)
logger.info(model.summary())
keras.utils.plot_model(model, "images/housing_model.png", show_shapes=True)


########## Training ##########
if do_training:
    logger.info("Training model:")
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid))
    logger.info("Training done.")


########## Plots ##########
if do_plots:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    fn.save_fig("keras_learning_curves_plot")
    plt.show()

