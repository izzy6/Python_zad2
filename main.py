import numpy as np
import re
import matplotlib.pyplot as plt
# import scipy as sp
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
    # string_array = array[:, 4:]
    # mode = sp.mode(string_array)
    # median_array = []
    # min_array = []
    # max_array = []
    # for a in array:
    #     # Get only float values
    #     float_array = a[:,:4].astype(float)
    #     # Median
    #     median_tmp = np.median(float_array, axis=0)
    #     median_array.append(median_tmp)
    #     # Minimum
    #     min_tmp = np.min(float_array, axis=0)
    #     min_array.append(min_tmp)
    #     # Miximum
    #     max_tmp = np.max(float_array, axis=0)
    #     max_array.append(max_tmp)
    print(median)
    print(min)
    print(max)
    # print(mode[0])

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

        elif (operation == '2'):
            data = read_database(database_path_2)
            print("base 2")
        elif (operation == '3'):
            data = read_database(database_path_1)
            print("base 3")
        elif(operation == '4'):
            is_finish = 1

