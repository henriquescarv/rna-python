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
X = data_csv[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']].values
y = data_csv['Performance Index'].values
y = normalize(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preparação dos dados (treinamento)
X_normalized = normalize(X_train)
y_normalized = y_train

# Configuração do modelo
input_size = X_normalized.shape[1]
hidden_size = 10
output_size = 1
learning_rate = 0.1
epochs = 600


# Treinamento
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
losses = nn.train(X_normalized, y_normalized, epochs, mse_loss, mse_loss_derivative)

# Avaliação do modelo
y_pred = nn.forward(X_normalized)
r2 = r2_score(y_normalized, y_pred)
print(f"Coeficiente de Determinação (R2): {r2}")

# Plotando o erro
plt.plot(losses)
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()
