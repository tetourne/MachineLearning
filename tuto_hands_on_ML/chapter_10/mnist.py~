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


do_plots = True
do_training = True


########## Configure logger ##########
logs_path = Path(__file__).parents[0] / "logs"
if not logs_path.is_dir():
    logs_path.mkdir(parents=True, exist_ok=True)
logs_file = logs_path / sys.argv[0].replace('.py', '.log')
logger = fn.my_logger(logs_file)


########## Load data ##########
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Split and scale data
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.


########## Parameters of DNN ##########
input_shape = [28,28]
n_hidden1 = 300
acti_hidden1 = 'relu'
n_hidden2 = 100
acti_hidden2 = 'relu'
n_outputs = 10
acti_outputs = 'softmax'


########## Define model ##########
logger.info("Defining model with")
logger.info(f"input shape: {input_shape} ; n_hidden1: {n_hidden1} ;\
              n_hidden2: {n_hidden2} ; n_outputs: {n_outputs}")

# model = keras.models.Sequential()
# # layers
# model.add(keras.layers.Flatten(input_shape=input_shape))
# model.add(keras.layers.Dense(n_hidden1, activation='relu'))
# model.add(keras.layers.Dense(n_hidden2, activation='relu'))
# model.add(keras.layers.Dense(n_outputs, activation='softmax'))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(n_hidden1, activation=acti_hidden1),
    keras.layers.Dense(n_hidden2, activation=acti_hidden2),
    keras.layers.Dense(n_outputs, activation=acti_outputs)
])


keras.backend.clear_session()  # releases the global state of keras
np.random.seed(42)
tf.random.set_seed(42)

# Defining loss and optimizer
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# summary of model
logger.info(model.layers)
logger.info(model.summary())
keras.utils.plot_model(model, "images/my_fashion_mnist_model.png", show_shapes=True)


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


########## Test model ##########
logger.info(model.evaluate(X_test, y_test))

# predictions
X_new = X_test[:3]
y_proba = model.predict(X_new)
logger.info(y_proba.round(2))
y_pred = np.argmax(model.predict(X_new), axis=-1)
logger.info(y_pred)
