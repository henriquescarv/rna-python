from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from rnn import categorical_crossentropy_loss, categorical_crossentropy_loss_derivative, NeuralNetwork

# Fetch dataset
zoo = fetch_ucirepo(id=111)

# Data (as pandas dataframes)
X = zoo.data.features  # Features
y = zoo.data.targets    # Target

# Aplicando One-Hot Encoding à coluna 'legs'
X = pd.get_dummies(X, columns=['legs'], prefix='legs', dtype=int)

# Convertendo y para One-Hot Encoding
y_onehot = pd.get_dummies(y, columns=['type'], prefix='type', dtype=int)

X = X.to_numpy()
y_onehot = y_onehot.to_numpy()

# Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
hidden_size = 10
output_size = y_train.shape[1]  # Número de classes
learning_rate = 0.01
epochs = 1000

# Inicializando a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, output_activation='softmax')

# Treinando a rede
losses = nn.train(X_train, y_train, epochs, categorical_crossentropy_loss, categorical_crossentropy_loss_derivative)

# Avaliação
y_pred = nn.predict(X_test, 'multiclass')
accuracy = np.mean(np.argmax(y_test, axis=1) == y_pred) * 100
print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")

plt.plot(losses)
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()
