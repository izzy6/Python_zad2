import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix  
 

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


data = readDataset()
x = data.drop('class', axis=1)
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20) 

svmclassifier = SVC(kernel='linear', max_iter=1)  
svmclassifier.fit(x_train, y_train)

y_pred = svmclassifier.predict(x_test)  

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
