from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
import pandas as pd

hda = pd.read_csv('dataset/heart_desease_ajustado.csv')
print('DataFrame Atualizada: \n',hda.head())

#Sem normalização do conjunto de dados, o dataset foi dividido em treino e teste.
print(hda["HeartDisease"].value_counts(), '\n')

X= hda.drop(['HeartDisease'], axis=1)
y=hda['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(   
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y)

print(y_train.value_counts())

#KNN implementado, exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
y_pred= knn.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)
