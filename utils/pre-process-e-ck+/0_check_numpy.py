import numpy as np

# Cargar el archivo .npy
data = np.load('data/S005_001.npy')

# Mostrar las primeras 10 filas
print(data[0])
print(data[-1])
