import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

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
    # Display values
    print("Median: {}".format(median))
    print("Minimum value: {}".format(min))
    print("Maximum value: {}".format(max))
    print("Mode: {}".format(mode))

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

def generate_histograms(array):
    view = plt.figure()
    pl1 = view.add_subplot(4,2,1)
    pl2 = view.add_subplot(4,2,5)
    pl1.hist(array[:, :1], bins=20, color='b', alpha=0.3)
    pl2.hist(array[:, 1:2], bins=20, color='b', alpha=0.3)
    plt.show()
# start
# static data
database_path_1 = 'database/iris.data'
database_path_2 = 'database/iris.data' #TODO change

is_finish = 0
while is_finish == 0:
    print("Select database:")
    print("1. Iris")
    print("2. ??")
    print("3. ??")

    operation = '0'
    while not re.match("^[1-4]", operation):
        operation = input()
    else:
        if (operation == '1'):
            data = read_database(database_path_1)
            iris_operation(data)
            generate_histograms(data)

        elif (operation == '2'):
            data = read_database(database_path_2)
            print("base 2")
        elif (operation == '3'):
            data = read_database(database_path_1)
            print("base 3")
        elif(operation == '4'):
            is_finish = 1

