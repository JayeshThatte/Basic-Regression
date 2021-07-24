# Import all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Read the dataset
f = pd.read_csv("Salary.csv")
exp = f["YearsExperience"].values
salary = f["Salary"].values

# Change 1D to 2D
exp = np.reshape(exp,(len(exp),1))
salary = np.reshape(salary,(len(salary),1))

# Since there is a large difference between exp and its
# corresponding salary, we need to scale down salary
y = StandardScaler()
salary_scaled = y.fit_transform(salary)

# Create the model , train it and predict
model = SVR()
model.fit(exp,salary_scaled)
salary_predictions = model.predict(exp)
salary_predictions = y.inverse_transform(model.predict(exp))

#to get intercept. Intercept is value of y at x=0
c = y.inverse_transform(model.predict([[0]]))

# Calculate the absolute mean error
toterror = 0
for i in range(len(salary_predictions)):
    toterror += abs(salary_predictions[i] - salary[i])
mean_error = toterror/len(salary_predictions)

# Provide insight into the model
print("Mean error is : ",mean_error)
print("X intercept is : ",c)

# Plot the graph
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title("SVM Regression")
plt.scatter(exp,salary)
plt.plot(exp,salary_predictions,color="red")
plt.show()



