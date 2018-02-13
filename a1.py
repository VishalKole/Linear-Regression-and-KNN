from pylab import *
#import matplotlib.pyplot as plt

"""
@Author - Vishal V Kole - vvk3025@rit.edu
For partial completion of the course Pattern Recognition
"""

"""
function to compute linear classifier
"""
def linear():
    #loading the data to work with
    data = load("data.npy")

    #prepping the data by removing the x and y values and the clas values
    x_vals = data[:, 0:2]
    second = x_vals[100:, :]
    first = x_vals[:100, :]
    temp = data[:, 2]
    y_vals = []

    #assigning class -1 and 1 so that we could equate the center to 0
    for item in temp:
        if (item == 0):
            y_vals.append(-1)
        else:
            y_vals.append(1)

    #creating the matrix and the transpose to apply the formulae
    complete_data = np.vstack((second, first))
    complete_data = np.array([np.ones(200), complete_data[:, 0], complete_data[:, 1]])
    complete_data = complete_data.T

    #creating the y matrix as in the formulae
    y = np.concatenate((y_vals[100:], y_vals[:100]))
    y = y.T

    #computing the beta matrix, thats the bias for the given training set
    computed_beta = np.linalg.inv(complete_data.T @ complete_data) @ (complete_data.T @ y)

    #create points for the line to plot
    line_x = np.linspace(-3, 4.8)

    #generate the Y coordinates WRT the x coordinates created above
    line_y = -computed_beta[0] / computed_beta[2] - (computed_beta[1] / computed_beta[2]) * line_x

    #plotting the given data
    i = 0
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        if (data[i, 2] == 1):
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='g')
        else:
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='b')

    #plotting the linear classification line
    plt.plot(line_x, line_y, color='black')

    v = axis()
    gap = 0.1

    #computing the vector to perform cross product to see the side to which the point lies
    vector1 = [line_x[-1] - line_x[0], line_y[-1] - line_y[0]]
    for x in np.arange(v[0], v[1], gap):
        for y in np.arange(v[2], v[3], gap):
            vector2 = [line_x[-1] - x, line_y[-1] - y]
            xp = vector1[0] * vector2[1] - vector1[1] * vector2[0]

            #check which side it falls
            if xp > 0:
                plt.scatter(x, y, marker='o', color='b', s=0.2)

            elif xp < 0:
                plt.scatter(x, y, marker='o', color='g', s=0.2)
    gc = 0
    bc = 0
    i = 0

    #computing the classification rate
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        vector2 = [line_x[-1] - data[i, 1], line_y[-1] - data[i, 0]]
        xp = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if ((xp >= 0) and (data[i, 2] == 1)):
            gc = gc + 1

        if ((xp < 0) and (data[i, 2] == 0.0)):
            bc = bc + 1

    #printing the results
    print("          Classification rate for linear classifier is - " + str(((gc + bc) / 200)*100) + "%")
    print("          Confusion matrix for linear classifier is as follows:")
    print("                   Predicted       Predicted")
    print("                   green           blue")
    print("   Actual Green     " + str(gc) + "              " + str(100 - gc) + "")
    print("   Actual Blue      " + str(100 - bc) + "              " + str(bc) + "")
    print();
    print()

    #saving the plot
    plt.savefig("linear.png")

"""
function to compute the K nearest neighbours with specific K
"""
def KNN(knn):

    #load the data to work with
    data = load("data.npy")
    gap = 0.03
    i = 0

    #plot the data points
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        if (data[i, 2] == 1):
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='g')
        else:
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='b')

    v = axis()
    meshgrid = {}

    #compute the class of current point
    for x in np.arange(v[0], v[1], gap):
        for y in np.arange(v[2], v[3], gap):
            if knn == 1:
                z = find_Kmin1(x, y, data)
            elif knn == 15:
                z = find_Kmin15(x, y, data)
            meshgrid[str(x) + str(y)] = [x, y, z]

    plt.clf()

    #assign a seperate class to the boundry point
    for x in np.arange(v[0], v[1], gap):
        last = 0
        for y in np.arange(v[2], v[3], gap):
            if (last != meshgrid.get(str(x) + str(y))[2]):
                last = meshgrid.get(str(x) + str(y))[2]
                meshgrid[(str(x) + str(y))] = [x, y, 2]

    for y in np.arange(v[2], v[3], gap):
        last = 1
        for x in np.arange(v[0], v[1], gap):
            if (last != meshgrid.get(str(x) + str(y))[2]):
                last = meshgrid.get(str(x) + str(y))[2]
                meshgrid[(str(x) + str(y))] = [x, y, 2]

    #plotting the boundry
    for key in meshgrid.items():
        if (key[1][2] == 2):
            plt.scatter(key[1][0], key[1][1], marker='o', color='black', s=1)

    i = 0
    #plotting the data
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        if (data[i, 2] == 1):
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='g', s=7)
        else:
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='b', s=7)

    #plotting the mesh
    for x in np.arange(v[0], v[1], gap * 4):
        for y in np.arange(v[2], v[3], gap * 4):
            if knn == 1:
                z = find_Kmin1(x, y, data)
            elif knn == 15:
                z = find_Kmin15(x, y, data)
            if (z == 1):
                plt.scatter(x, y, marker='o', color='g', s=0.5)
            else:
                plt.scatter(x, y, marker='o', color='b', s=0.5)
    gc = 0
    bc = 0
    i = 0

    #computing the classification rate
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        if knn == 1:
            if find_Kmin1(data[i, 0], data[i, 1], data) == 1 and data[i, 2] == 1:
                gc = gc + 1
            if find_Kmin1(data[i, 0], data[i, 1], data) == 0 and data[i, 2] == 0.0:
                bc = bc + 1
        if knn == 15:
            if find_Kmin15(data[i, 0], data[i, 1], data) == 1 and data[i, 2] == 1:
                gc = gc + 1
            if find_Kmin15(data[i, 0], data[i, 1], data) == 0 and data[i, 2] == 0.0:
                bc = bc + 1

    #printing the results
    print("          Classification rate for KNN classifier with K="+ str(knn) +" is - " + str(((gc + bc) / 200)*100) + "%")
    print("          Confusion matrix for linear classifier is as follows:")
    print("                   Predicted       Predicted")
    print("                   green           blue")
    print("   Actual Green     " + str(gc) + "              " + str(100 - gc) + "")
    print("   Actual Blue      " + str(100 - bc) + "              " + str(bc) + "")
    print();
    print()
    if knn == 1:
        plt.savefig("KNN1.png")
    elif knn == 15:
        plt.savefig("KNN15.png")

"""
 function to find the class 
"""
def find_Kmin1(x, y, data):
    all_dist = []
    for item in data:
        all_dist.append([fastest_calc_dist([x, y], item), item[2]])
        list.sort(all_dist)
    return all_dist[0][1]

#function to find the euclidean distance
def fastest_calc_dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

"""
 function to find the class 
"""
def find_Kmin15(x, y, data):
    all_dist = []
    for item in data:
        all_dist.append([fastest_calc_dist([x, y], item), item[2]])
        list.sort(all_dist)
    sum = 0
    for i in range(15):
        sum = sum + all_dist[i][1]
    sum = sum / 15
    if (sum >= 0.5):
        sum = 1
    else:
        sum = 0
    return sum


if __name__ == "__main__":
    linear()
    KNN(1)
    KNN(15)
