# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o dataset Iris
iris = load_iris()
data = iris.data
target = iris.target

# Criando um DataFrame com o Pandas
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(data, columns=columns)
df['species'] = pd.Categorical.from_codes(target, iris.target_names)

# Dividindo os dados em treino e teste
X = data
y = target

# Visualizações
# 1. Pair plot
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species')
plt.show()

# 2. Matriz de correlação
plt.figure(figsize=(8, 6))
correlation_matrix = df.drop('species', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()
