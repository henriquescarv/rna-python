from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rnn import categorical_crossentropy_loss, categorical_crossentropy_loss_derivative, NeuralNetwork

# Carregar o dataset do zoologico
zoo = fetch_ucirepo(id=111)

# Selecionar variáveis independentes (X) e dependentes (y)
X = zoo.data.features 
y = zoo.data.targets   

# Aplicando One-Hot Encoding
X = pd.get_dummies(X, columns=['legs'], prefix='legs', dtype=int)
y_onehot = pd.get_dummies(y, columns=['type'], prefix='type', dtype=int)

X = X.to_numpy()
y_onehot = y_onehot.to_numpy()

# Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Definindo hiperparâmetros
input_size = X_train.shape[1]
print(input_size)
hidden_size = 11 # Camada oculta
output_size = y_train.shape[1]  # Número de classes
learning_rate = 0.01 # Taxa de aprendizado
epochs = 55 # Épocas

# Inicializando a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, output_activation='softmax')

# Treinando a rede
train_losses, test_losses = nn.train(X_train, y_train, epochs, 
                                     categorical_crossentropy_loss, 
                                     categorical_crossentropy_loss_derivative, 
                                     X_test=X_test, y_test=y_test)

# Avaliação dos dados de treinamento
y_pred_train = nn.predict(X_train, 'multiclass')
accuracy_train = np.mean(np.argmax(y_train, axis=1) == y_pred_train) * 100
print(f"Acurácia no conjunto de dados de treinamento: {accuracy_train:.2f}%")

# Avaliação dos dados de teste
y_pred_test = nn.predict(X_test, 'multiclass')
accuracy_test = np.mean(np.argmax(y_test, axis=1) == y_pred_test) * 100
print(f"Acurácia no conjunto de dados de teste: {accuracy_test:.2f}%")

# Verificar se a acurácia de teste é maior que 50% para passar na pipeline
assert accuracy_test > 50, f'Acurácia abaixo de 50% no teste: {accuracy_test}'

# Gráfico das perdas
plt.plot(train_losses, label="Treinamento")
plt.plot(test_losses, label="Teste")
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.legend()
plt.show()
