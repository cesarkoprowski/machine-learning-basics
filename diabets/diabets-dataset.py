# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.datasets import load_diabetes

# Carregando o dataset Diabetes
diabetes = load_diabetes()

# Criando o DataFrame
df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Adicionando a coluna alvo (progressão da doença)
df_diabetes['progression'] = diabetes.target

# Exibindo informações sobre o dataset
print("\nInformações do Dataset Diabetes:")
print(df_diabetes.info())

print("\nPrimeiras 10 linhas do Dataset Diabetes:")
print(df_diabetes.head(10))

print("\nEstatísticas descritivas do alvo (progressão):")
print(df_diabetes['progression'].describe())