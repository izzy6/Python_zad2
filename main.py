import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

def read_database(database):
    df = pd.read_csv(database, sep=',')
    data = df.values
    # data_array = []
    # # Split to 3 arrays
    # data_1 = data[data[:, 4] == 'Iris-setosa', :]
    # data_2 = data[data[:, 4] == 'Iris-versicolor', :]
    # data_3 = data[data[:, 4] == 'Iris-virginica', :]
    # data_array.append(data_1)
    # data_array.append(data_2)
    # data_array.append(data_3)
    # return data_array
    return data

def iris_operation(array):
    float_array = array[:, :4].astype(float)
    # Median
    median = np.median(float_array, axis=0)
    # Minimum
    min = np.min(float_array, axis=0)
    # Miximum
    max = np.max(float_array, axis=0)
    # # Mode
    string_array = array[:, 4:]
    trans_array = np.transpose( string_array )
    count_1 = ['Iris-setosa' , np.count_nonzero( trans_array == 'Iris-setosa')]
    count_2 = ['Iris-versicolor', np.count_nonzero(trans_array == 'Iris-versicolor')]
    count_3 = ['Iris-virginica', np.count_nonzero(trans_array == 'Iris-virginica')]
    count_list_int = [count_1[1], count_2[1], count_3[1]]
    count_list = [count_1, count_2, count_3]
    mode = []
    for x in count_list:
        if x[1] == np.max(count_list_int):
            mode.append(x[0])
    print(count_list_int)
    # Display values
    print("Median: {}".format(median))
    print("Minimum value: {}".format(min))
    print("Maximum value: {}".format(max))
    print("Mode: {}".format(mode))

def generate_iris_histograms(array):
    data_1 = []
    data_2 = []
    data_3 = []
    for a in array:
        if a[4] == 'Iris-setosa':
            data_1.append(a)
        elif a[4] == 'Iris-versicolor':
            data_2.append(a)
        elif a[4] == 'Iris-virginica':
            data_3.append(a)
    #print(data_1)
    array_1 = np.array([data_1[0], data_1[1], data_1[2], data_1[3]])
    array_2 = np.array([data_2[0], data_2[1], data_2[2], data_2[3]])
    array_3 = np.array([data_3[0], data_3[1], data_3[2], data_3[3]])

    view = plt.figure()


    pl1 = view.add_subplot(4, 3, 1)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 15])
    pl2 = view.add_subplot(2, 3, 2)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 15])
    pl3 = view.add_subplot(2, 3, 3)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 15])
    pl4 = view.add_subplot(2, 3, 4)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 50])
    pl5 = view.add_subplot(2, 3, 5)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 50])
    pl6 = view.add_subplot(2, 3, 6)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 50])

    pl1.hist([array[:, :1], array_1[:, :1]], bins=20, color=['b','r'], alpha=0.7)
    pl2.hist([array[:, :1], array_2[:, :1]], bins=20, color=['b', 'r'], alpha=0.7)
    pl3.hist([array[:, :1], array_3[:, :1]], bins=20, color=['b', 'r'], alpha=0.7)

    pl4.hist([array[:, :2], array_1[:, :2]], bins=20, color=['b','r'], alpha=0.7)
    pl5.hist([array[:, :2], array_2[:, :2]], bins=20, color=['b', 'r'], alpha=0.7)
    pl6.hist([array[:, :2], array_3[:, :2]], bins=20, color=['b', 'r'], alpha=0.7)
    plt.show()

def births_operation(array):
    number_of_births = array[:,2]
    number_of_births_mean = number_of_births.mean()
    print("Mean of births per day: %d" % number_of_births_mean)
def generate_births_histograms(array):
    plt.plot(np.arange(10))

# start
# static data
database_path_1 = 'database/iris.data'
database_path_2 = 'database/births.csv'

is_finish = 0
while is_finish == 0:
    print("Choose option:")
    print("1. For mark 3 - Iris")
    print("2. For mark 4 - Births")
    print("3. Terminate program")

    operation = '0'
    while not re.match("^[1-3]", operation):
        operation = input()
    else:
        if (operation == '1'):
            data = read_database(database_path_1)
            iris_operation(data)
            generate_iris_histograms(data)
        elif (operation == '2'):
            data = read_database(database_path_2)
            births_operation(data)
            generate_births_histograms(data)
        elif (operation == '3'):
            is_finish = 1           

