import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import keras
from keras import Sequential

data = pd.read_csv("dataset/ProcessedData.csv")

# Splitting data into training and testing set
x = data.drop(columns=["stroke"])
y = data["stroke"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)

# Logistic Regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
print("Logistic Regression accuracy: ", accuracy_score(y_test, y_pred_lr))

# Random Forest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("Random Forest accuracy: ", accuracy_score(y_test, y_pred_rf))

# Neural network model
nn: Sequential = keras.models.load_model('neural_network/best_model.keras')
y_pred_nn = nn.predict(x_test)
y_pred_nn = np.argmax(y_pred_nn, axis=1)
print("Neural Network accuracy", accuracy_score(y_test, y_pred_nn))