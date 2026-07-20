from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler
import pandas as pd
import numpy as np

hda = pd.read_csv('dataset/heart_desease_ajustado.csv')

#Normalização do conjunto de dados com normalização logarítmica e verificação da acurácia do knn.
X = hda.drop('HeartDisease', axis=1)
y = hda['HeartDisease']

#Divisão do conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y)


#Normalização L2 
X_train_l2=normalize(X_train, norm='l2')
X_test_l2=normalize(X_test, norm='l2')  

knn_l2 = KNeighborsClassifier()
knn_l2.fit(X_train_l2, y_train)

y_pred_l2 = knn_l2.predict(X_test_l2)
acuracia_l2 = accuracy_score(y_test, y_pred_l2)

'''
X_norm_log = np.log1p(X)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_norm_log, y, test_size=0.3, random_state=42, stratify=y)

knn = KNeighborsClassifier()
knn.fit(X_train_l2, y_train_log)

y_predlog = knn.predict(X_test_log)
acuracialog = accuracy_score(y_test_log, y_predlog)'''

#Normalização do conjunto de dados com normalização de media zero(0) e variância unitária(1) e verificação a acurácia do knn.
scaler = StandardScaler()
scaler.fit(X_train) #aprendendo apenas com os dados de treino para evitar vazamento de dados

#Aplicando a transformação
X_train_std  = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

knn_std = KNeighborsClassifier()
knn_std.fit(X_train_std, y_train)
y_pred_std = knn_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)

#Print das duas acuracias lado a lado para comparar.
print('Acurácia com normalização de log: {:.2f}%'.format(acuracia_l2 * 100))
print('Acurácia com normalização de média zero e variância unitária: {:.2f}%'.format(acc_std * 100))
