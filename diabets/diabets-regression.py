# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset diabetes
diabetes = load_diabetes()
X = diabetes.data[:, 2].reshape(-1, 1)  # Usando apenas uma feature para visualização
y = diabetes.target

# Visualizando os dados originais
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados Originais')
plt.xlabel('BMI (Índice de Massa Corporal)')
plt.ylabel('Progressão da Diabetes')
plt.title('Dados Originais do Dataset Diabetes')
plt.legend()
plt.show()

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = modelo.predict(X_test)

# Calculando métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMétricas de Avaliação:")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Criando 10 novos exemplos para previsão
novos_dados = np.linspace(X.min(), X.max(), 10).reshape(-1, 1)
previsoes = modelo.predict(novos_dados)

print("\nPrevisões para 10 novos exemplos:")
for i, (x, pred) in enumerate(zip(novos_dados, previsoes), 1):
    print(f"Exemplo {i}: BMI = {x[0]:.2f}, Previsão = {pred:.2f}")

# Visualizando os resultados finais
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados Originais')
plt.plot(X_test, y_pred, color='red', label='Linha de Regressão')
plt.scatter(novos_dados, previsoes, color='green', marker='*',
           s=200, label='Novas Previsões')
plt.xlabel('BMI (Índice de Massa Corporal)')
plt.ylabel('Progressão da Diabetes')
plt.title('Regressão Linear com Novas Previsões')
plt.legend()
plt.show()
