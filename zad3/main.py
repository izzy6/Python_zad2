import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
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

def fillMissingValuesWithHotDeck(dataset, datasetWithoutNan):
    print("\nZastepowanie brakujacych wartosci  wartościami najbliższymi (odleglosc euklidesowa) na podstawie najbardziej skorelowanej cechy..")
    tmp = []
    for d in dataset.values:
        if np.isnan(d[3]) :
            dist = []
            for dwn in datasetWithoutNan.values:
                dist.append(distance.euclidean(d[5], dwn[5]))
            d[3] = datasetWithoutNan.values[dist.index(min(dist)), 3]
        tmp.append(d[3])
    dataset['horsepower'].fillna(tmp[0], inplace = True)
    return dataset

def fillMissingValuesFromRegressionLine(dataset, coef):
    print("\nZastepowanie brakujacych wartosci  wartościami na podstawie wzoru na funkcję regresji..")
    tmp = []
    for d in dataset.values:
        if np.isnan(d[3]) :
            d[3] = coef[0] + coef[1] * d[5]
        tmp.append(d[3])
    dataset['horsepower'].fillna(tmp[0], inplace = True)
    return dataset

def fillMissingValuesWithInterpolation(dataset):
    print("\nZastepowanie brakujacych wartosci poprzez wartosci po interpolacji...")
    new_dataset = dataset.copy()
    new_dataset.set_index('acceleration', inplace = True)
    new_dataset.sort_index(inplace = True)
    new_dataset['horsepower'].interpolate(method='index', inplace = True)
    print(new_dataset.head(20))
    return new_dataset


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

def clearValues(dataset, modValue):
    tmp = []
    for d in range(len(dataset.values)):
        if d % modValue == 0 :
            dataset.loc[d , 'horsepower'] = np.nan
    return dataset


def estimate_coef(x, y): 
    #wzory z internetu do wyliczenia krzywej regresji (chyba o to chodzi)
    #y = a + b*x
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    x_mean, y_mean = np.mean(x), np.mean(y) 
    #print("x_mean = {}, y_mean = {}, n = {}".format(x_mean, y_mean, n))
    # calculating cross-deviation and deviation about x 
    # SS_xy = np.sum(y*x - n*y_mean*x_mean)
    # SS_xx = np.sum(x*x - n*x_mean*x_mean)
  
    # calculating regression coefficients 
    # b = SS_xy / SS_xx
    b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    a = y_mean - b*x_mean 
  
    return(a, b) 


def plot_regression_line(x, y, coef): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector
    y_pred = coef[0] + coef[1]*x 
  
    # regression line 
    plt.plot(x, y_pred, color = "g") 
  
    plt.xlabel('horsepower') 
    plt.ylabel('acceleration')
    plt.show()

def subplot_regression_line(x, y, coef, color):
        # plotting the actual points as scatter plot
        plt.scatter(x, y, color=color,
                    marker="o", s=30)

        # predicted response vector
        y_pred = coef[0] + coef[1] * x
        # regression line
        plt.plot(x, y_pred, color='k')

        plt.xlabel('horsepower')
        plt.ylabel('acceleration')
        return plt


def printStatistics(dataset):
    print("\n\n------- STATYSTYKA ZBIORU DANYCH --------")
    x = np.asarray(dataset['horsepower'])
    st_dev = statistics.stdev(x)
    x_mean = np.mean(x)
    n = np.size(x)

    print("odchylenie standardowe = {}, \nśrednia = {}, \nliczba rekordów = {}\n\n".format(st_dev, x_mean, n))

def countForDifferentDatasets(dataset, color):
    datasetWithDeletedRows = deleteRowsWithMissingValues(dataset)
    datasetAfterHotDock = fillMissingValuesWithHotDeck(dataset, datasetWithDeletedRows)

    coefAfter = estimate_coef(np.asarray(datasetAfterHotDock['horsepower']).reshape(-1, 1),
                              np.asarray(datasetAfterHotDock['acceleration']).reshape(-1, 1))
    print("Wyznaczone współczynniki regresji po imputacji:\na = {} \nb = {}".format(coefAfter[0], coefAfter[1]))
    printStatistics(datasetAfterHotDock)
    return subplot_regression_line(np.asarray(datasetAfterHotDock['horsepower']).reshape(-1, 1),
                         np.asarray(datasetAfterHotDock['acceleration']).reshape(-1, 1), coefAfter, color)


###### THE END OF FUNCTIONS
# 3
dataset = readDataset()

countMissingValues(dataset)
datasetWithNan = dataset.copy()
datasetWithDeletedRows = deleteRowsWithMissingValues(dataset)

#we have to pass array parameters of columns that we want to use for our regression function
coefBefore = estimate_coef(np.asarray(datasetWithDeletedRows['horsepower']).reshape(-1,1), np.asarray(datasetWithDeletedRows['acceleration']).reshape(-1,1))
print("Wyznaczone współczynniki regresji przed imputacją:\na = {} \nb = {}".format(coefBefore[0], coefBefore[1]))
printStatistics(datasetWithDeletedRows)

#--------------------------------------
print("Metoda uzupełniania średnimi wartościami")
#data imputation with means
datasetWithFilledMeanValues = fillMissingValuesWithMeans(dataset)

regression_function = plot_regression_line(np.asarray(datasetWithDeletedRows['horsepower']).reshape(-1,1), np.asarray(datasetWithDeletedRows['acceleration']).reshape(-1,1), coefBefore)

coefAfter = estimate_coef(np.asarray(datasetWithFilledMeanValues['horsepower']).reshape(-1,1), np.asarray(datasetWithFilledMeanValues['acceleration']).reshape(-1,1))
print("Wyznaczone współczynniki regresji po imputacji:\na = {} \nb = {}".format(coefAfter[0], coefAfter[1]))
printStatistics(datasetWithFilledMeanValues)


print("Wnioski: Współczynniki krzywej regresji zmieniły się nieznacznie ponieważ w drugim przypadku badany zbiór danych jest większy o 23 rekordy. \nNatomiast średnia wartość kolumn nie zmieniła się ponieważ do wypełnienia brakujących danych zostały wykorzystane średnie wartości")
print("Po imputacji danych wychylenie standardowe nieznacznie się zmniejszyło.\n")
plot_regression_line(np.asarray(datasetWithFilledMeanValues['horsepower']).reshape(-1,1), np.asarray(datasetWithFilledMeanValues['acceleration']).reshape(-1,1), coefAfter)
#
# #4
# #--------------------------------------
dataset = datasetWithNan.copy() # dataset with nan
# print("Metoda Hot Dock")
datasetAfterHotDock = fillMissingValuesWithHotDeck(dataset, datasetWithDeletedRows)
plot_regression_line(np.asarray(datasetWithDeletedRows['horsepower']).reshape(-1,1), np.asarray(datasetWithDeletedRows['acceleration']).reshape(-1,1), coefBefore)

coefAfter = estimate_coef(np.asarray(datasetAfterHotDock['horsepower']).reshape(-1,1), np.asarray(datasetAfterHotDock['acceleration']).reshape(-1,1))
print("Wyznaczone współczynniki regresji po imputacji:\na = {} \nb = {}".format(coefAfter[0], coefAfter[1]))
printStatistics(datasetAfterHotDock)
print("Wnioski: ??")
plot_regression_line(np.asarray(datasetAfterHotDock['horsepower']).reshape(-1,1), np.asarray(datasetAfterHotDock['acceleration']).reshape(-1,1), coefAfter)

#-------------------------------------
dataset = datasetWithNan.copy() # dataset with nan
print("Metoda interpolacji")

fillMissingValuesWithInterpolation = fillMissingValuesWithInterpolation(dataset)

coefAfter = estimate_coef(np.asarray(fillMissingValuesWithInterpolation['horsepower']).reshape(-1,1), np.asarray(fillMissingValuesWithInterpolation.index).reshape(-1,1))
print("Wyznaczone współczynniki regresji po imputacji:\na = {} \nb = {}".format(coefAfter[0], coefAfter[1]))
printStatistics(fillMissingValuesWithInterpolation)

print("Wnioski: ")
plot_regression_line(np.asarray(fillMissingValuesWithInterpolation['horsepower']).reshape(-1,1), np.asarray(fillMissingValuesWithInterpolation.index).reshape(-1,1), coefAfter)

#-------------------------------------
dataset = datasetWithNan.copy() # dataset with nan
print("Metoda wykorzystująca krzywą regresji")

fillMissingValuesFromRegressionLine = fillMissingValuesFromRegressionLine(dataset, coefAfter)

coefAfter = estimate_coef(np.asarray(fillMissingValuesFromRegressionLine['horsepower']).reshape(-1,1), np.asarray(fillMissingValuesFromRegressionLine['acceleration']).reshape(-1,1))
print("Wyznaczone współczynniki regresji po imputacji:\na = {} \nb = {}".format(coefAfter[0], coefAfter[1]))
printStatistics(fillMissingValuesFromRegressionLine)

print("Wnioski: ")
plot_regression_line(np.asarray(fillMissingValuesFromRegressionLine['horsepower']).reshape(-1,1), np.asarray(fillMissingValuesFromRegressionLine['acceleration']).reshape(-1,1), coefAfter)

#5
view = plt.figure()

# ------ 15%
dataset_15 = clearValues(datasetAfterHotDock.copy(), 7)
countMissingValues(dataset_15)
print("Metoda Hot Dock dla 14% brakujacych wartości")
pl1 = view.add_subplot(3, 1, 1)
plt.title("Dla 14% pustych wartości")
pl1 = countForDifferentDatasets(dataset_15, 'm')
#-------30%
dataset_30 = clearValues(datasetAfterHotDock.copy(), 3)
countMissingValues(dataset_30)
print("Metoda Hot Dock dla 33% brakujacych wartości")
pl2 = view.add_subplot(3, 1, 2)
plt.title("Dla 33% pustych wartości")
pl2 = countForDifferentDatasets(dataset_30, 'g')
# --------- 45%
dataset_45 = clearValues(datasetAfterHotDock.copy(), 2)
countMissingValues(dataset_45)
print("Metoda Hot Dock dla 50% brakujacych wartości")
pl3 = view.add_subplot(3, 1, 3)
plt.title("Dla 50% pustych wartości")
pl3 = countForDifferentDatasets(dataset_45, 'r')
plt.tight_layout()
plt.show()
# WNIOSKI!!!!!

