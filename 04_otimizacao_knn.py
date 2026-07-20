import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

hda = pd.read_csv('dataset/heart_desease_ajustado.csv')

#Normalização com a melhor normalização o conjunto de dados se houver melhoria.
X = hda.drop('HeartDisease', axis=1)
y = hda['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y)

# Padronização dos dados com média zero e variância unitária
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Testar vários valores de K
neighbors = np.arange(1, 12)

train_accuracies = []
test_accuracies = []

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)

    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))


#Melhor valor de K
best_index = np.argmax(test_accuracies)
best_k = list(neighbors)[best_index]
best_accuracy = test_accuracies[best_index]
print(f"Melhor K: {best_k}")
print(f"Acurácia: {best_accuracy:.4f}")


#Gráfico com o a indicação do melhor k.
plt.figure(figsize=(8,5))
plt.plot(neighbors, train_accuracies, marker="o", label="Treino")
plt.plot(neighbors, test_accuracies, marker="o", label="Teste")
plt.scatter(best_k, best_accuracy, s=100, label=f"Melhor K = {best_k}")
plt.xlabel("Número de vizinhos (K)")
plt.ylabel("Acurácia")
plt.title("Escolha do melhor K para o KNN")
plt.grid(True)
plt.legend()
plt.show()
