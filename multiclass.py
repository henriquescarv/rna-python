import numpy as np
import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from rnn import relu, relu_derivative

# Carregar o dataset
data_path = kagglehub.dataset_download('kukuroo3/body-performance-data')
dataset = os.listdir(data_path)[0]
data_csv = pd.read_csv(os.path.join(data_path, dataset))
data_csv['gender'] = data_csv['gender'].map({'M': 0, 'F': 1})
data_csv['class'] = data_csv['class'].map({'D': 0, 'C': 1, 'B': 2, 'A': 3})

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

x = data_csv.drop(columns=['class']).values
y = data_csv['class'].values

x = normalize(x)
y = normalize(y.reshape(-1, 1))

