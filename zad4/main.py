import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
import matplotlib.pyplot as plt
import scipy.stats.contingency as cn
 

def readDataset():
    data = pd.read_csv('database/iris.data', names = ['swidth', 'slength', 'pwidth', 'plength', 'class'])

    #change type of data in car2 dataset
    data["swidth"] = pd.to_numeric(data["swidth"])
    data["slength"] = pd.to_numeric(data["slength"])
    data["pwidth"] = pd.to_numeric(data["pwidth"])
    data["plength"] = pd.to_numeric(data["plength"])
    return data

def drawPlot(scores_test, scores_train):
    plt.plot(scores_train, color='green', alpha=0.8, label='Przed redukcją')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Po redukcji')
    plt.title("Dokładność w zależności od l. iteracji", fontsize=14)
    plt.xlabel('Iteracje')
    plt.legend(loc='upper left')
    plt.show()

def classification(data):
    #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
    x = data.drop('class', axis=1)
    y = data['class']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.20)

    iterations = 50
    scores_train = []
    scores_test = []

    for i in range(iterations):
        clf = SGDClassifier(n_iter=i)
        #clf.partial_fit(x_train, y_train, classes=np.unique(y_train))
        clf.fit(x_train, y_train)
        scores_train.append(clf.score(x_train, y_train))
        scores_test.append(clf.score(x_test, y_test))

    #drawPlot(scores_test, scores_train)
    return scores_test

def classificationAfterChanges(data, clasifiedData):
    #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
    x = data.drop('class', axis=1)
    y = data['class']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.20)

    iterations = 50
    scores_train = []
    scores_test = []

    for i in range(iterations):
        clf = SGDClassifier(n_iter=i)
        #clf.partial_fit(x_train, y_train, classes=np.unique(y_train))
        clf.fit(x_train, y_train)
        scores_train.append(clf.score(x_train, y_train))
        scores_test.append(clf.score(x_test, y_test))
    drawPlot(scores_test, clasifiedData)

def pcaReduction(data):
    x = data.drop('class', axis=1)
    y = data['class']

    # Standardizing the features - bez skalowania algorytm PCA może nie działać
    x = StandardScaler().fit_transform(x)

    #redukacja z 4 wymiarów(atrybutów) do 2
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    dataAfterReduction = pd.DataFrame(data = principalComponents, columns = ['komponent 1', 'komponent 2'])
    #print(dataAfterReduction)
    reducedDataWithClasses = pd.concat([dataAfterReduction, y], axis = 1)
    #print(dataWithClasses)
    
    #wyswietla dane na wykresie po redukcji
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = reducedDataWithClasses['class'] == target
        ax.scatter(reducedDataWithClasses.loc[indicesToKeep, 'komponent 1']
                , reducedDataWithClasses.loc[indicesToKeep, 'komponent 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()
    return reducedDataWithClasses

def test_independence(data, colX, colY):
    X = data[colX].astype(str)
    Y = data[colY].astype(str)
    observed = pd.crosstab(Y,X)
    chi2, p, dof, expected = cn.chi2_contingency(observed.values)
    return p

import warnings
warnings.simplefilter("ignore")
#3
data = readDataset()
dataClear = data.copy()
trainData = classification(data)
reducedData = pcaReduction(data)
classificationAfterChanges(reducedData,  trainData)
#print('Po redukcji klasyfikacja jest jakby dokładniejsza')

#4
# Wariancja
dataStd = dataClear.std(axis=0, ddof=1)
dataVar = np.power(dataStd, 2)
print(dataVar)
print("Najmniejszą wariancję mają cechy: slength i plenght")
dataMinVar = pd.DataFrame(dataClear, columns=['slength', 'plength', 'class'])
classificationAfterChanges(dataMinVar, trainData)
# Test niezależności chi**2
dataChi = []
labels = list(dataClear)
for i in range(4):
    for j in range(4):
        if(i != j):
            dataChi.append([labels[i], labels[j], test_independence(dataClear, labels[i], labels[j])])
minChi = sorted(dataChi, key=lambda x: x[2])[0]
print('Najmniejszą zależność({}) na podstawie testu chi**2 mają cechy: {} i {}'.format(minChi[2], minChi[0], minChi[1]))
dataMinChi = pd.DataFrame(dataClear, columns=[ minChi[0],  minChi[1], 'class'])
classificationAfterChanges(dataMinChi,  trainData)




