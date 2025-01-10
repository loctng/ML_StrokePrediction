import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
  
def removeOutliers(data, features):
  # Visualize outliers with boxplot
  for i, feature in enumerate(features, 1):
    plt.subplot(1,len(features),i)
    data[[feature]].boxplot()
  plt.show()
  
  # Find the index of observations to be removed
  removeIdx = pd.Series([False] * len(data))
  for feature in features:
    # Find the min and max value range of each feature
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    min = q1 - 1.5 * (q3 - q1)
    max = q3 + 1.5 * (q3 - q1)
    
    outliers_lower = data[feature] < min
    outliers_upper = data[feature] > max
    
    removeIdx = removeIdx | outliers_lower | outliers_upper

  return data.loc[~removeIdx]


data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Clean missing values, which only appear in BMI column
noBMI = np.isnan(data["bmi"])
data = data.loc[~noBMI]
data = data.reset_index(drop=True)

# Remove outliers
data = removeOutliers(data, ["age", "avg_glucose_level", "bmi"])
data = data.reset_index(drop=True)

# Drop unnecessary features
data = data.drop(columns=["id", "work_type"])

# One-hot-encoding some features
data = pd.get_dummies(data, columns=["gender", "ever_married", "Residence_type", "smoking_status"], dtype=int)
data = data.drop(columns=["gender_Other", "ever_married_No", "Residence_type_Rural", "smoking_status_Unknown"])
data = data.rename(
  columns={"ever_married_Yes": "married", 
           "Residence_type_Urban": "residence_urban",
           "smoking_status_formerly smoked": "smoking_formerly",
           "smoking_status_never smoked": "smoking_never",
           "smoking_status_smokes": "smoking_yes"
          }
)

# Standardize numerical features
numerical_features = ["age", "avg_glucose_level", "bmi"]
numerical_values = data[numerical_features]
scaler = StandardScaler()
numerical_values = scaler.fit_transform(numerical_values)
numerical_values = pd.DataFrame(numerical_values)
numerical_values.columns = numerical_features

data = data.drop(columns=numerical_features)
data = pd.concat([data, numerical_values], axis=1)

data.to_csv("ProcessedData.csv", index=False)