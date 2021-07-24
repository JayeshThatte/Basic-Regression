# Import all the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the dataset
f = pd.read_csv("Salary.csv")
exp = f["YearsExperience"].values
salary = f["Salary"].values

# Change 1D to 2D
exp = np.reshape(exp,(len(exp),1))
salary = np.reshape(salary,(len(salary),1))

# Create the model , train it and predict
model = LinearRegression()
model.fit(exp,salary)
salary_predict = model.predict(exp)

#to get intercept. Intercept is value of y at x=0
c = model.predict([[0]])

# Calculate the absolute mean error
toterror = 0
for i in range(len(salary_predict)):
    toterror += abs(salary_predict[i] - salary[i])
mean_error = toterror/len(salary_predict)

# Provide insight into the model
print("Mean error is : ",mean_error)
print("X intercept is : ",c)
print("Weight of model is : ",model.coef_)

# Plot the graph
plt.title("Linear Regression")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(exp,salary)
plt.plot(exp,salary_predict,color='red')
plt.show()