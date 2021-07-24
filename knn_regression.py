# Import all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.neighbors import  KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Read the dataset
f = pd.read_csv("Salary.csv")
exp = f["YearsExperience"].values
salary = f["Salary"].values

# Change 1D to 2D
exp = np.reshape(exp,(len(exp),1))
salary = np.reshape(salary,(len(salary),1))

# Create the model , train it and predict
model = KNeighborsRegressor()
model.fit(exp,salary)
salary_predictions = model.predict(exp)

# Calculate the absolute mean error
toterror = 0
for i in range(len(salary_predictions)):
    toterror += abs(salary_predictions[i] - salary[i])
mean_error = toterror/len(salary_predictions)

# Provide insight into the model
print("Mean error is : ",mean_error)

# Plot the graph
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title("KNN Regression")
plt.scatter(exp,salary)
plt.plot(exp,salary_predictions,color="red")
plt.show()
