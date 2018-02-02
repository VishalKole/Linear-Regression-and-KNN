import numpy as np
from pylab import *
import matplotlib.pyplot as plt

data = load("data.npy")

# Read in data
x_vals = data[: ,0:2]
temp = data[:,2]
y_vals = []

for item in temp:
    if (item == 0):
        y_vals.append(-1)
    else:
        y_vals.append(1)


# Throw out one data point so that there
# are an equal number from each class.
class_one = x_vals[100:, :]
class_two = x_vals[:100, :]


# Create complete data array comprised
# of all points from both classes.
X = np.vstack((class_one, class_two))

# Add column of ones to account for bias term
X = np.array([np.ones(200), X[:, 0], X[:, 1]]).T

# Create y array of class labels
y = np.concatenate((y_vals[100:], y_vals[:100])).T

# Calculate the Regularized Least Squares solution
beta = np.linalg.inv(X.T @ X) @ (X.T @ y)



line_x = np.linspace(-3, 5.1)
line_y = -beta[0] / beta[2] - (beta[1] / beta[2]) * line_x



i=0
while(i<len(data[:,1])-1):
    i=i+1
    if(data[i,2]==1):
        plt.scatter(data[i, 0], data[i, 1], marker='o', color='g')
    else:
        plt.scatter(data[i,0],data[i,1],marker='o',color='b')

v = axis()
gap=0.1
for x in np.arange(v[0],v[1], gap):
   for y in np.arange(v[2],v[3], gap):
       plt.scatter(x,y,marker='o',color='r' ,s=0.2)

plt.plot(line_x, line_y)


plt.show()



