import pandas as pd 
import numpy as np
import sklearn.model_selection

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load the data
df = pd.read_csv('student.csv')

# Assuming df is your DataFrame
df = pd.get_dummies(df)

# Now, you can split your data and fit the model
X = df.drop('G3', axis=1)  # features
y = df['G3']  # target

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeRegressor and fit it to the training data
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = reg.predict(X_test)

# Print the mean squared error
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

# Print the Root Mean Squared Error
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Print the Mean Absolute Error
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))

# Print the R-squared score
print("R-squared score: ", r2_score(y_test, y_pred))

scores = cross_val_score(reg, X, y, cv=10, scoring='neg_mean_squared_error')

# Take a square root and make scores positive
rmse_scores = np.sqrt(-scores)

# Print the mean RMSE score
print("Mean RMSE: ", rmse_scores.mean())