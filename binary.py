import numpy as np
import kagglehub
import os
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from rnn import binary_crossentropy_loss, binary_crossentropy_loss_derivative, NeuralNetwork

# Carregar o dataset do câncer
cancer_dataset_path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
cancer_dataset = os.listdir(cancer_dataset_path)[0]
cancer_csv = pandas.read_csv(os.path.join(cancer_dataset_path, cancer_dataset))

# Função para normalizar os dados
def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Preparar os dados
df = cancer_csv.drop(columns=["id"])  # Remover coluna irrelevante
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # Mapear M/B para 1/0

# Selecionar variáveis independentes (X) e dependentes (y)
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values.reshape(-1, 1)

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
hidden_size = 10 # Camada oculta
output_size = y_train_normalized.shape[1] # Número de classes
learning_rate = 0.01 # Taxa de aprendizado
epochs = 100 # Épocas

# Inicializando a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, output_activation='sigmoid')

# Treinando a rede
train_losses, test_losses = nn.train(X_train_normalized, y_train_normalized, epochs, 
                                     binary_crossentropy_loss, binary_crossentropy_loss_derivative,
                                     X_test=X_test_normalized, y_test=y_test_normalized)

# Avaliação dos dados de treinamento
y_pred_train = nn.predict(X_train_normalized, 'binary')
accuracy_train = np.mean(y_pred_train == y_train_normalized) * 100
print(f"Acurácia no conjunto de dados de treinamento: {accuracy_train:.2f}%")

# Avaliação dos dados de teste
y_pred_test = nn.predict(X_test_normalized, 'binary')
accuracy_test = np.mean(y_pred_test == y_test_normalized) * 100
print(f"Acurácia no conjunto de dados de teste: {accuracy_test:.2f}%")

# Gráfico das perdas
plt.plot(train_losses, label="Treinamento")
plt.plot(test_losses, label="Teste")
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.legend()
plt.show()