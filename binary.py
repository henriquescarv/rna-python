import numpy as np
import kagglehub
import os
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from rnn import binary_crossentropy_loss, binary_crossentropy_loss_derivative, NeuralNetwork

# Carregar o dataset
cancer_dataset_path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
cancer_dataset = os.listdir(cancer_dataset_path)[0]
cancer_csv = pandas.read_csv(os.path.join(cancer_dataset_path, cancer_dataset))

# Carregar o dataset
df = cancer_csv.drop(columns=["id"])  # Remover coluna irrelevante
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # Mapear M/B para 1/0

# Selecionar variáveis independentes (X) e dependentes (y)
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para normalizar os dados
def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Normalizar os dados de entrada (treinamento)
X_normalized = normalize(X_train)
y = y_train

input_size = X_normalized.shape[1]
hidden_size = 16  # Você pode ajustar esse valor
output_size = 1
learning_rate = 0.001
epochs = 3000

# Instanciar e treinar a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, output_activation='sigmoid')
losses = nn.train(X_normalized, y, epochs, binary_crossentropy_loss, binary_crossentropy_loss_derivative)

# Prever os dados de teste
y_pred = nn.predict(X_normalized)

# Avaliação da rede
accuracy = np.mean(y_pred == y) * 100
print(f"Acurácia no conjunto de dados: {accuracy:.2f}%")

plt.plot(losses)
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()