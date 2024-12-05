import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from rnn import mse_loss, mse_loss_derivative, r2_score, NeuralNetwork

# Carregar o dataset de performance de estudantes
data_path = kagglehub.dataset_download('nikhil7280/student-performance-multiple-linear-regression')
dataset = os.listdir(data_path)[0]
data_csv = pd.read_csv(os.path.join(data_path, dataset))

# Função para normalizar os dados
def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Preparar os dados
X = data_csv[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
X = X.copy()
X['Extracurricular Activities'] = X['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = X.values
y = data_csv['Performance Index'].values
y = normalize(y.reshape(-1, 1))

# Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preparação dos dados (treinamento)
X_train_normalized = normalize(X_train)
y_train_normalized = y_train

# Preparação dos dados (teste)
X_test_normalized = normalize(X_test)
y_test_normalized = y_test

# Definindo hiperparâmetros
input_size = X_train_normalized.shape[1]
hidden_size = 20 # Camada oculta
output_size = 1 # Número de classes
learning_rate = 0.1 # Taxa de aprendizado
epochs = 50 # Épocas

# Inicializando a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# Treinando a rede
train_losses, test_losses = nn.train(X_train_normalized, y_train_normalized, epochs, 
                                     mse_loss, mse_loss_derivative, 
                                     X_test=X_test_normalized, y_test=y_test_normalized)

# Avaliação dos dados de treinamento
y_train_pred = nn.forward(X_train_normalized)
r2_train = r2_score(y_train_normalized, y_train_pred)
print(f"Coeficiente de Determinação (R2) - treinamento: {r2_train}")

# Avaliação dos dados de teste
y_test_pred = nn.forward(X_test_normalized)
r2_test = r2_score(y_test_normalized, y_test_pred)
print(f"Coeficiente de Determinação (R2) - teste: {r2_test}")

# Verificar se R2 de teste é maior que 0.5 para passar na pipeline
assert r2_test > 0.5, f'R2 abaixo de 0.5'

# Gráfico das perdas
plt.plot(train_losses, label="Treinamento")
plt.plot(test_losses, label="Teste")
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.legend()
plt.show()
