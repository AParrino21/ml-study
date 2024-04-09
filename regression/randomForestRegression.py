# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# importing the dataset
dataset = pd.read_csv('/Users/anthonyparrino/code/ml-study/dummyData/Position_Salaries.csv')
# setting all data besides first column to X
X = dataset.iloc[:, 1:-1].values
# setting all data in last column (dependant variable) to y
y = dataset.iloc[:, -1].values

# training the random forest regression model on the whole dataset, passing in the number of trees to create
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X, y)

# predicting a new salary from a postion level 6.5
predicted_salary = regressor.predict([[6.5]])
print(predicted_salary)

# visualising the decision tree regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()