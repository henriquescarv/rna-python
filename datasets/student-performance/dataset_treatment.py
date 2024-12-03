import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import kagglehub

# Diretório local para armazenar os datasets
output_dir = "../student-performance"
os.makedirs(output_dir, exist_ok=True)

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Baixar o dataset usando kagglehub
data_path = kagglehub.dataset_download('nikhil7280/student-performance-multiple-linear-regression')
dataset = os.listdir(data_path)[0]
data = pd.read_csv(os.path.join(data_path, dataset))
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
data['Performance Index'] = normalize(data['Performance Index'].values.reshape(-1, 1))

# Dividir em treino (80%) e teste (20%)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Salvar os arquivos divididos em CSVs
train_path = os.path.join(output_dir, "student_performance_train.csv")
test_path = os.path.join(output_dir, "student_performance_test.csv")

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f"Arquivos salvos:\nTreinamento: {train_path}\nTeste: {test_path}")