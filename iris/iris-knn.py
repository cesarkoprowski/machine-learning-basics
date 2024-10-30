import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Carregando o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Selecionando as duas melhores características usando SelectKBest
selector = SelectKBest(f_classif, k=2)
X_best = selector.fit_transform(X_scaled, y)
selected_features = selector.get_support()

# Identificando quais características foram selecionadas
feature_names = np.array(iris.feature_names)
selected_feature_names = feature_names[selected_features]
print(f"\nCaracterísticas selecionadas: {selected_feature_names}\n\n")

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Avaliando o modelo
y_pred = knn.predict(X_test)

# Calculando e plotando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Imprimindo o relatório de classificação
print(f"\nAcurácia do modelo: {accuracy_score(y_test, y_pred):.2f}")

# Calculando os centróides de cada classe
centroids = []
for i in range(3):
    mask = y == i
    centroid = np.mean(X_best[mask], axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)

#------------------------------------------------------------------------------
# Gerando 20 novos pontos próximos aos centróides
np.random.seed(42)
new_samples = []
new_labels = []

for i in range(20):
    class_idx = i % 3
    centroid = centroids[class_idx]
    noise = np.random.normal(0, 0.3, size=2)
    new_point = centroid + noise
    new_samples.append(new_point)
    new_labels.append(class_idx)

new_samples = np.array(new_samples)
new_labels = np.array(new_labels)

# Fazendo previsões para os novos pontos
new_predictions = knn.predict(new_samples)

# Plotando os resultados finais
plt.figure(figsize=(12, 8))

# Plotando os dados originais do algoritmo
scatter = plt.scatter(X_best[:, 0], X_best[:, 1], c=y, cmap='viridis', alpha=0.6, label='Dados Originais')

# Plotando os centróides
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centróides')

# Plotando os novos pontos
plt.scatter(new_samples[:, 0], new_samples[:, 1], c=new_labels, cmap='viridis',
           marker='*', s=200, edgecolor='black', label='Novos Pontos')

plt.colorbar(scatter)
plt.title('Visualização Final com Novos Pontos e Centróides')
plt.xlabel(selected_feature_names[0])
plt.ylabel(selected_feature_names[1])
plt.legend()
plt.grid(True)
plt.show()

# Mostrando as coordenadas e classes dos novos pontos
print("\nDetalhes dos novos pontos gerados:")
print("\nFormato: [Coordenada 1, Coordenada 2] -> Classe")
for i, (sample, pred) in enumerate(zip(new_samples, new_labels)):
    print(f"Ponto {i+1}: {sample.round(3)} -> {iris.target_names[pred]}")

#------------------------------------------------------------------------------
# Criando dados de exemplo
data_example = {
    'sepal length (cm)': [5.0, 5.5, 4.9, 6.0, 5.5, 6.5, 7.0, 6.5, 7.5],
    'sepal width (cm)': [3.5, 3.8, 3.0, 2.8, 3.0, 3.2, 3.0, 3.5, 3.0],
    'petal length (cm)': [1.4, 1.5, 1.3, 4.5, 4.0, 4.7, 6.0, 5.5, 6.5],
    'petal width (cm)': [0.2, 0.3, 0.2, 1.3, 1.2, 1.5, 2.0, 1.8, 2.0],
    'species': ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica']
}

# Convertendo os dados de exemplo para DataFrame
df_example = pd.DataFrame(data_example)

# Plotando os dados de exemplo em um gráfico separado usando petal length e petal width
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_example, x='petal length (cm)', y='petal width (cm)', hue='species', style='species', s=100)
plt.title('Scatter Plot de Exemplo de Dimensões de Pétala para Espécies de Iris')
plt.xlabel('Comprimento da Pétala (cm)')
plt.ylabel('Largura da Pétala (cm)')
plt.legend(title='Espécies')
plt.grid(True)
plt.show()