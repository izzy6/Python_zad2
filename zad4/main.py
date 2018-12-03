import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection  
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
import matplotlib.pyplot as plt
 

def readDataset():
    data = pd.read_csv('database/iris.data', names = ['swidth', 'slength', 'pwidth', 'plength', 'class'])
    #print(data)

    #change type of data in car2 dataset
    data["swidth"] = pd.to_numeric(data["swidth"])
    data["slength"] = pd.to_numeric(data["slength"])
    data["pwidth"] = pd.to_numeric(data["pwidth"])
    data["plength"] = pd.to_numeric(data["plength"])
    #print(data)
    return data

def drawPlot(scores_test, scores_train):
    plt.plot(scores_train, color='green', alpha=0.8, label='Trening')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Dokładność w zależności od l. iteracji", fontsize=14)
    plt.xlabel('Iteracje')
    plt.legend(loc='upper left')
    plt.show()

#na podstawie https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
def classification(data):
    #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
    x = data.drop('class', axis=1)
    y = data['class']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.20) 

    clf = SGDClassifier()

    iterations = 50
    scores_train = []
    scores_test = []

    for i in range(iterations):
        #clf.partial_fit(x_train, y_train, classes=np.unique(y_train))
        clf.fit(x_train, y_train)
        scores_train.append(clf.score(x_train, y_train))
        scores_test.append(clf.score(x_test, y_test))

    drawPlot(scores_test, scores_train)


data = readDataset()
classification(data)


