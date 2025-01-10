import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential, layers, losses
import keras_tuner


data = pd.read_csv("ProcessedData.csv")

# Splitting data into training and testing set
x = data.drop(columns=["stroke"])
y = data["stroke"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=27)


def nnModel(hp):
  model = Sequential()
  model.add(layers.Input(shape=(12,)))
  
  # Hidden layers
  for i in range(hp.Int('num_layers', 1, 2)):
    model.add(
      layers.Dense(
        units=hp.Choice('units', [16, 32]), 
        activation=hp.Choice('activation', ['relu', 'tanh'])
      )
    )
    
  model.add(layers.Dense(2, "sigmoid"))
  model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
  return model

# Grid search
tuner = keras_tuner.GridSearch(
  hypermodel=nnModel,
  objective='val_accuracy',
  overwrite="true",
  directory="neural_network",
  project_name="history"
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

models = tuner.get_best_models()
best_model = models[0]
best_model.summary()

best_model.save('neural_network/best_model.keras')