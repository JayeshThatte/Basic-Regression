# Import all the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read the dataset
f = pd.read_csv("Salary.csv")
exp = f["YearsExperience"].values
salary = f["Salary"].values

# Change 1D to 2D
exp = np.reshape(exp,(len(exp),1))
salary = np.reshape(salary,(len(salary),1))

# Set degree 2 for the polynomial equation
# 3 over fit , 1 underfit so use degree 2
polynomial_features = PolynomialFeatures(degree=2)
exp_poly = polynomial_features.fit_transform(exp)

# Create the model , train it and predict
model = LinearRegression()
model.fit(exp_poly, salary)
salary_predictions = model.predict(exp_poly)

#to get intercept. Intercept is value of y at x=0
c = polynomial_features.fit_transform([[0]])
c = model.predict(c)

# Calculate the absolute mean error
toterror = 0
for i in range(len(salary_predictions)):
    toterror += abs(salary_predictions[i] - salary[i])
mean_error = toterror/len(salary_predictions)

# Provide insight into the model
print("Mean error is : ",mean_error)
print("X intercept is : ",c)
print("Weight of model is : ",model.coef_)

# Plot the graph
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title("Polynomial Regression Degree 2")
plt.scatter(exp,salary)
plt.plot(exp,salary_predictions,color="red")
plt.show()
