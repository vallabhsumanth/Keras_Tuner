# Keras_Tuner

This project features a  learning model optimized using Keras Tuner, which automates the search for the best hyperparameters. By leveraging advanced optimization techniques, the model achieves superior performance on the Fashion MNIST dataset compared to baseline configurations.KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. Easily configure your search space with a define-by-run syntax, then leverage one of the available search algorithms to find the best hyperparameter values for your models. KerasTuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms built-in, and is also designed to be easy for researchers to extend in order to experiment with new search algorithms.

# Installation
pip install keras-tuner --upgrade

# Import KerasTuner and TensorFlow:

import keras_tuner
import keras

# Write a function that creates and returns a Keras model. Useing the hp argument to define

def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1, activation='relu'))
  model.compile(loss='mse')
  return model
  
# Initialize a tuner for RandomSearch. We use objective to specify the objective to select the best models, and we use max_trials to specify the number of different models to try.

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
    
# Start the search and get the best model:

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
