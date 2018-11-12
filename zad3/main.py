import pandas as pd
import numpy

def readDataset():
    data = pd.read_csv('cars.csv', names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'])
    #replace ? missing values with python NaN - values with a NaN value are ignored from operations like sum, count
    data = data.replace('?', numpy.NaN)
    return data

def countMissingValues(dataset):
    
    print("Brakujące dane według kolumn")
    print("row_num        missing_val")
    print(dataset.isnull().sum())
    #print(dataset.head(20))
    nan_count = 0
    for index, row in dataset.iterrows():
        if (row.isna().any()):
            nan_count += 1
    print("Liczba rekordów z co najmniej jedną brakującą wartością: %d" % nan_count)
    percentage = nan_count*100/dataset.shape[0]
    print("Procent rekordów z co najmniej jedną brakującą wartością: %d %%" % percentage)

def getDatasetWithoutMissingValues(dataset):
    print("Kasowanie rekordów zawierających braki...")
    print("Rozmiar bazy przed kasowaniem:")
    print(dataset.shape)
    dataset = dataset.dropna(axis=0, how='any')
    #print(dataset.head(20))
    print("Rozmiar bazy po skasowaniu:")
    print(dataset.shape)
    return dataset
dataset = readDataset()
countMissingValues(dataset)
datasetWithoutMissingVal = getDatasetWithoutMissingValues(dataset)


