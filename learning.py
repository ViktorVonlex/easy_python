import pandas as pd 
import numpy as np
import sklearn.model_selection

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load the data
df = pd.read_csv('student.csv')

# Assuming df is your DataFrame
df = pd.get_dummies(df)

# Now, you can split your data and fit the model
X = df.drop('G3', axis=1)  # features
y = df['G3']  # target

# Add a new column with the average alcohol consumption
df['AvgAlc'] = (df['Dalc'] + df['Walc']) / 2

# Fill missing values with the mean
df = df.fillna(df.mean())

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeRegressor and fit it to the training data
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = reg.predict(X_test)

# Print the mean squared error
print("DecTree Mean Squared Error: ", mean_squared_error(y_test, y_pred))

# Print the Root Mean Squared Error
print("DecTree Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Print the Mean Absolute Error
print("DecTree Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))

# Print the R-squared score
print("DecTree R-squared score: ", r2_score(y_test, y_pred))

# Create a RandomForestRegressor and fit it to the training data
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

# Predict the labels for the test data
y_pred_rf = rf_reg.predict(X_test)

# Print the mean squared error for the Random Forest model
print("Random Forest Mean Squared Error: ", mean_squared_error(y_test, y_pred_rf))

# Perform 10-fold cross validation for both models
scores_dt = cross_val_score(reg, X, y, cv=10, scoring='neg_mean_squared_error')
scores_rf = cross_val_score(rf_reg, X, y, cv=10, scoring='neg_mean_squared_error')

# Take a square root and make scores positive
rmse_scores_dt = np.sqrt(-scores_dt)
rmse_scores_rf = np.sqrt(-scores_rf)

# Print the mean RMSE score for both models
print("Decision Tree Mean RMSE: ", rmse_scores_dt.mean())
print("Random Forest Mean RMSE: ", rmse_scores_rf.mean())


# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

# Create a RandomForestRegressor
rf_reg = RandomForestRegressor()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train a new RandomForestRegressor with the best parameters
best_rf_reg = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
best_rf_reg.fit(X_train, y_train)

# Predict the labels for the test data
y_pred_best_rf = best_rf_reg.predict(X_test)

# Perform 5-fold cross validation for the best Random Forest model
scores_best_rf = cross_val_score(best_rf_reg, X, y, cv=5, scoring='neg_mean_squared_error')

# Take a square root and make scores positive
rmse_scores_best_rf = np.sqrt(-scores_best_rf)

# Print the mean RMSE score for the best Random Forest model
print("Best Random Forest Mean RMSE: ", rmse_scores_best_rf.mean())

# Print the mean squared error for the best Random Forest model
print("Best Random Forest Mean Squared Error: ", mean_squared_error(y_test, y_pred_best_rf))

# Print the R-squared score for the best Random Forest model
print("Best Random Forest R-squared Score: ", r2_score(y_test, y_pred_best_rf))

# Perform cross-validation
scores = cross_val_score(best_rf_reg, X, y, cv=5)

# Print the mean cross-validation score
print("Cross-validated Score: ", scores.mean())

