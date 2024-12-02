import numpy as np

# Funções de ativação (ReLU para camadas ocultas, as outras para a saída)
def relu(z):
    return np.maximum(0, z)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Limita valores extremos p/ evitar overflow
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)


# Funções de perda
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_true.size

def binary_crossentropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true


# Função para calcular o coeficiente de determinação (R2) para o problema de regressão
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, output_activation='linear'):
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        
        # Inicialização dos pesos e bias
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        # Propagação para frente
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output

        if (self.output_activation == 'linear'):
          self.final_output = linear(self.final_input)

        if (self.output_activation == 'sigmoid'):
            self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, y_pred, loss_derivative):
        # Erro na saída
        if (self.output_activation == 'linear'):
          output_error = loss_derivative(y, y_pred) * linear_derivative(self.final_input)
        
        if (self.output_activation == 'sigmoid'):
          output_error = loss_derivative(y, y_pred) * sigmoid_derivative(self.final_input)
        
        # Gradientes para pesos e bias da saída
        weights_hidden_output_grad = np.dot(self.hidden_output.T, output_error)
        bias_output_grad = np.sum(output_error, axis=0, keepdims=True)

        # Erro na camada oculta
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * relu_derivative(self.hidden_input)
        
        # Gradientes para pesos e bias da camada oculta
        weights_input_hidden_grad = np.dot(X.T, hidden_error)
        bias_hidden_grad = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Atualização dos pesos e biases
        self.weights_hidden_output -= self.learning_rate * weights_hidden_output_grad
        self.bias_output -= self.learning_rate * bias_output_grad

        self.weights_input_hidden -= self.learning_rate * weights_input_hidden_grad
        self.bias_hidden -= self.learning_rate * bias_hidden_grad

    def train(self, X, y, epochs, loss_function, loss_derivative):
        losses = []
        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)
            
            # Cálculo do erro
            loss = loss_function(y, y_pred)
            losses.append(loss)

            # Backward
            self.backward(X, y, y_pred, loss_derivative)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return losses
    
    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)  # Classificação binária