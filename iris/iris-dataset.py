# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.datasets import load_iris

# Carregando o dataset Iris
iris = load_iris()

# Criando o DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Adicionando a coluna de espécies
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Exibindo informações sobre o dataset
print("\nInformações do Dataset:")
print(df.info())

print("\nPrimeiras 10 linhas do Dataset:")
print(df.head(10))

print("\nContagem de cada espécie:")
print(df['species'].value_counts())