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
    view = plt.figure()
    pl1 = view.add_subplot(1, 1, 1)
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 60])

    pl1.hist([array[:, 2], array[:, 3]], bins=10, color=['b','r'], alpha=0.5, label=['petal length', 'petal width'])
    pl1.set_title('Histogram of petal length and width')
    pl1.set_xlabel('Number of irises')
    pl1.set_ylabel('[cm]')
    pl1.legend()
    plt.show()

def births_operation(array):
    births = array[:,2]
    births_mean = births.mean()
    print("Mean of births per day: %d" % births_mean)
    return births_mean

def generate_births_histograms(array, births_mean):

    bins = np.linspace(np.min(array[:,2]), np.max(array[:,2]), 20)
    births = np.asarray(array[:,2], dtype=np.float64)
    plt.hist(births, bins=bins, color='orange')
    plt.xlabel("Number of births per day")
    plt.ylabel("Number of days with number of births in given range")
    plt.title("Histogram of births per day")
    plt.axvline(x=births_mean, color='black', linestyle='--')
    plt.text(births_mean,500,'Mean of births',rotation=90)
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

