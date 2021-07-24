# Import all the required libraries
import random
import  matplotlib.pyplot as plt
import  pandas as pd

# Read the dataset
f = pd.read_csv("Salary.csv")
exp = f["YearsExperience"].values
salary = f["Salary"].values

# Get random weight
INIT = random.uniform(-30,30)

# Since we dont know the learning rate, we check weights for each learning rate
learning_rt = 0.000000
wts = []
rates = []
bias = 20000

# Check weight for each learning rate 
for i in range(100000):
    learning_rt+=0.0000001

    wt = float(INIT)
    for i in range(len(exp)):
        pred = (exp[i] * wt) + bias
        wt = wt -( learning_rt * 2 * (pred - salary[i]) * exp[i])
    wts += [wt]
    rates += [learning_rt]

# Plot graph of Rates v/s Weights
plt.xlabel('Rates')
plt.ylabel('Weights')
plt.title("Rates v/s Weights")
plt.plot(rates,wts,color="red")
plt.show()

# Pick a learning rate from graph where weight is a constant
learning_rt = 0.001217
wt = float(INIT)
for i in range(len(exp)):
        pred = (exp[i] * wt)+ bias
        wt = wt -( learning_rt * 2 * (pred - salary[i]) * exp[i])

# Actually predict the model
salary_predictions = []
for i in range(len(exp)):
    pred = (exp[i] * wt) + bias
    salary_predictions += [pred]
print("Weight is : ",wt)

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
plt.title("Neural Network")
plt.scatter(exp,salary)
plt.plot(exp,salary_predictions,color="red")
plt.show()

