import pandas as pd
import numpy

def readDataset():
    data = pd.read_csv('cars.csv', names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'])
    #replace ? missing values with python NaN - values with a NaN value are ignored from operations like sum, count
    data = data.replace('?', numpy.NaN)

    #trzeba zamienic niektore kolumny w ktorych brakuje danych na typ numeryczny zeby nie powodowaly problemow w dalszej czesci programu
    data["normalized-losses"] = pd.to_numeric(data["normalized-losses"])
    data["horsepower"] = pd.to_numeric(data["horsepower"])
    data["peak-rpm"] = pd.to_numeric(data["peak-rpm"])
    data["price"] = pd.to_numeric(data["price"])
    return data

def printMissingValues(dataset):
    print("\nBrakujące dane według kolumn")
    print("\nrow_num        missing_val")
    print(dataset.isnull().sum())

def countMissingValues(dataset):
    
    printMissingValues(dataset)
    #print(dataset.head(20))
    nan_count = 0
    for index, row in dataset.iterrows():
        if (row.isna().any()):
            nan_count += 1
    print("\nLiczba rekordów z co najmniej jedną brakującą wartością: %d" % nan_count)
    percentage = nan_count*100/dataset.shape[0]
    print("Procent rekordów z co najmniej jedną brakującą wartością: %d %%" % percentage)

def deleteRowsWithMissingValues(dataset):
    print("\nKasowanie rekordów zawierających braki...")
    print("\nRozmiar bazy przed kasowaniem:")
    print(dataset.shape)
    dataset.dropna(axis=0, how='any', inplace=True)
    #print(dataset.head(20))
    print("\nRozmiar bazy po skasowaniu:")
    print(dataset.shape)
    #print(dataset.isnull().sum())
    return dataset

def fillMissingValuesWithMeans(dataset, datasetWithDeletedRows):
    print("\nZastepowanie brakujacych wartosci srednimi z kolumn...")
    dataset.fillna(dataset.mean(), inplace=True)
    printMissingValues(dataset)
    print(dataset.head(20))
    #print(dataset.mean())
    return dataset

def regressionFunction(dataset):
    #TODO: 3. Wyznaczyć krzywą regresji dla danych bez braków.
    print("To be continued...")

def compareDatasetsBeforeAndAfterImputation(datasetBefore, datasetAfter):
    #TODO: 5. Porównać charakterystykę zbiorów przed i po imputacji (średnia, odchylenie standardowe, kwartyle).
    print("To be continued...")

def regressionFunctionAfterImputation():
    #TODO: 6. Wyznaczyć krzywą regresji dla danych po imputacji. Porównać jak zmieniły się parametry krzywej.
    print("To be continued...")
    
dataset = readDataset()

countMissingValues(dataset)

datasetWithDeletedRows = deleteRowsWithMissingValues(dataset)

datasetWithFilledMeanValues = fillMissingValuesWithMeans(dataset, datasetWithDeletedRows)




