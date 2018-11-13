import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

# WAŻNE!!!!
# NIE WIEM CZY DOBRZE ZROZUMIAŁAM TO ZADANIE, ALE WSZYSTKIE OPERACJE ROBIE WLASCIWIE DLA KONKRETNEJ KOLUMNY W KTOREJ WYSTEPUJA BRAKI - "HORSEPOWER"
# W POZOSTALYCH KOLUMNACH NIE MA BRAKUJACYCH WARTOSCI


def readDataset():
    data = pd.read_csv('cars2.csv', names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

    #names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    #'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
    #'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'])

    #replace ? missing values with python NaN - values with a NaN value are ignored from operations like sum, count and "?" would cause problems
    data = data.replace('?', np.NaN)    
    print(data)
    #trzeba zamienic niektore kolumny w ktorych brakuje danych na typ numeryczny zeby nie powodowaly problemow w dalszej czesci programu

    #change type of data in car2 dataset
    data["horsepower"] = pd.to_numeric(data["horsepower"])
    data["acceleration"] = pd.to_numeric(data["acceleration"])

    #change type of data in car dataset
    #data["normalized-losses"] = pd.to_numeric(data["normalized-losses"])
    #data["horsepower"] = pd.to_numeric(data["horsepower"])
    #data["peak-rpm"] = pd.to_numeric(data["peak-rpm"])
    #data["price"] = pd.to_numeric(data["price"])
    #data["bore"] = pd.to_numeric(data["bore"])
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
    changedDataset = dataset.dropna(axis=0, how='any')
    #print(dataset.head(20))
    print("\nRozmiar bazy po skasowaniu:")
    print(dataset.shape)
    #print(dataset.isnull().sum())
    return changedDataset

def fillMissingValuesWithMeans(dataset):
    print("\nZastepowanie brakujacych wartosci srednimi z kolumn...")
    dataset.fillna(dataset.mean(), inplace=True)
    printMissingValues(dataset)
    print(dataset.head(20))
    #print(dataset.mean())
    return dataset

def regressionFunction(dataset):
    #rysowanie wykresow na podstawie funkcji regresji wyliczonej za pomoca metod z pakietu sklearn ale chyba nie o to tu jednak choodzi.. na wszelki wypadek nie kasuje funkcji
    #data feature
    x = np.asarray(dataset['horsepower']).reshape(-1,1)
    x_train = x[:-20]
    x_test = x[-20:]
    print(x_train)
    #print(x_test)
    
    #predicted values
    y = np.asarray(dataset['acceleration']).reshape(-1,1)
    y_train = y[:-20]
    y_test = y[-20:]
    print(y_train)

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    #plt.scatter(x, y,  color='red')

    print('Coefficients: \n', regr.coef_)

    # The mean square error
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x_test, y_test))
    
    # Plot outputs
    plt.scatter(x_test, y_test,  color='red')
    plt.plot(x_test, regr.predict(x_test), color='blue', linewidth=1)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def estimate_coef(x, y): 
    #wzory z internetu do wyliczenia krzywej regresji (chyba o to chodzi)
    #y = a + b*x
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    x_mean, y_mean = np.mean(x), np.mean(y) 
    #print("x_mean = {}, y_mean = {}, n = {}".format(x_mean, y_mean, n))
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x - n*y_mean*x_mean) 
    SS_xx = np.sum(x*x - n*x_mean*x_mean) 
  
    # calculating regression coefficients 
    b = SS_xy / SS_xx 
    a = y_mean - b*x_mean 
  
    return(a, b) 


def plot_regression_line(x, y, coef): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector - chyba stąd trzeba będzie uzyskać dane do imputacji w zadaniu na 4
    y_pred = coef[0] + coef[1]*x 
  
    # regression line 
    plt.plot(x, y_pred, color = "g") 
  
    plt.xlabel('horsepower') 
    plt.ylabel('acceleration') 
  
    plt.show() 


def printStatistics(dataset):
    print("\n\n------- STATYSTYKA ZBIORU DANYCH --------")
    x = np.asarray(dataset['horsepower'])
    st_dev = statistics.stdev(x)
    x_mean = np.mean(x)
    n = np.size(x)

    print("odchylenie standardowe = {}, \nśrednia = {}, \nliczba rekordów = {}\n\n".format(st_dev, x_mean, n))

####### THE END OF FUNCTIONS
    
dataset = readDataset()

countMissingValues(dataset)

datasetWithDeletedRows = deleteRowsWithMissingValues(dataset)

#we have to pass array parameters of columns that we want to use for our regression function
coefBefore = estimate_coef(np.asarray(datasetWithDeletedRows['horsepower']).reshape(-1,1), np.asarray(datasetWithDeletedRows['acceleration']).reshape(-1,1)) 
print("Wyznaczone współczynniki regresji przed imputacją:\na = {} \nb = {}".format(coefBefore[0], coefBefore[1])) 
printStatistics(datasetWithDeletedRows)
#data imputation with means
datasetWithFilledMeanValues = fillMissingValuesWithMeans(dataset)


plot_regression_line(np.asarray(datasetWithDeletedRows['horsepower']).reshape(-1,1), np.asarray(datasetWithDeletedRows['acceleration']).reshape(-1,1), coefBefore)

coefAfter = estimate_coef(np.asarray(datasetWithFilledMeanValues['horsepower']).reshape(-1,1), np.asarray(datasetWithFilledMeanValues['acceleration']).reshape(-1,1)) 
print("Wyznaczone współczynniki regresji po imputacji:\na = {} \nb = {}".format(coefAfter[0], coefAfter[1])) 
printStatistics(datasetWithFilledMeanValues)


print("Współczynniki krzywej regresji zmieniły się nieznacznie ponieważ w drugim przypadku badany zbiór danych jest większy o 23 rekordy. \nNatomiast średnia wartość kolumn nie zmieniła się ponieważ do wypełnienia brakujących danych zostały wykorzystane średnie wartości")
print("Po imputacji danych wychylenie standardowe nieznacznie się zmniejszyło.\n")
plot_regression_line(np.asarray(datasetWithFilledMeanValues['horsepower']).reshape(-1,1), np.asarray(datasetWithFilledMeanValues['acceleration']).reshape(-1,1), coefAfter)



