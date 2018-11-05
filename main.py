import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

def read_database(database):
    df = pd.read_csv(database, sep=',')
    data = df.values
    # data_array = []
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
    # data_1 = []
    # data_2 = []
    # data_3 = []
    # for a in array:
    #     if a[4] == 'Iris-setosa':
    #         data_1.append(a)
    #     elif a[4] == 'Iris-versicolor':
    #         data_2.append(a)
    #     elif a[4] == 'Iris-virginica':
    #         data_3.append(a)
    #
    # array_1 = np.array([data_1[0], data_1[1], data_1[2], data_1[3]])
    # array_2 = np.array([data_2[0], data_2[1], data_2[2], data_2[3]])
    # array_3 = np.array([data_3[0], data_3[1], data_3[2], data_3[3]])

    view = plt.figure()
    pl1 = view.add_subplot(1, 1, 1)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 60])
    # pl2 = view.add_subplot(2, 2, 2)
    # axes = plt.gca()
    # axes.set_xlim([0, 8])
    # axes.set_ylim([0, 10])
    # pl3 = view.add_subplot(2, 2, 3)
    # axes = plt.gca()
    # axes.set_xlim([0, 8])
    # axes.set_ylim([0, 10])
    # pl4 = view.add_subplot(2, 2, 4)
    # axes = plt.gca()
    # axes.set_xlim([0, 8])
    # axes.set_ylim([0, 10])
    pl1.hist([array[:, 2], array[:, 3]], bins=10, color=['b','r'], alpha=0.5, label=['petal length', 'petal width'])
    pl1.set_title('Histogram of petal length and width')
    pl1.set_xlabel('Number of irises')
    pl1.set_ylabel('[cm]')
    pl1.legend()
    # pl2.hist([array_1[:, 2], array_1[:, 3]], bins=5, color=['b', 'r'], alpha=0.5)
    # pl3.hist([array_2[:, 2], array_2[:, 3]], bins=5, color=['b', 'r'], alpha=0.5)
    # pl4.hist([array_3[:, 2], array_3[:, 3]], bins=5, color=['b', 'r'], alpha=0.5)
    plt.show()

def births_operation(array):
    births = array[:,2]
    births_mean = births.mean()
    print("Mean of births per day: %d" % births_mean)
    return births_mean

def generate_births_histograms(array, births_mean):
    births = array[:500,2]
    births = np.array(births)
    #dates = np.array(dates)
    print(births)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.scatter(dates, births, s=1)
    #ax.set_title('Histogram')
    #ax.legend(loc='upper left')
    #ax.set_ylabel('Number of births')
    #ax.set_xlim(xmin=dates[0], xmax=dates[-1])
    #fig.tight_layout
    #plt.axhline(y=births_mean, color='black', linewidth=1.5, label='Mean')
    #print(x)

    #x = np.random.random_integers(1, 100, 5)
    #print(x)
    #plt.hist(x, bins=20)
    #plt.ylabel('No of times')
    #plt.show()
    bin_edges = [8000, 8400, 8800, 9200, 9600, 10000, 10400, 10800, 11200, 11600]
    plt.hist(births, bins=bin_edges)
    plt.show()

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
            mean = births_operation(data)
            generate_births_histograms(data, mean)
        elif (operation == '3'):
            is_finish = 1           

