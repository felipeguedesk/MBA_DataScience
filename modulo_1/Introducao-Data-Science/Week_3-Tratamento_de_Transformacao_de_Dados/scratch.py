# gerador de números aleatórios
import random # gerador de números aleatórios
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt # biblioteca para visualização dos dados
from sklearn.preprocessing import StandardScaler # modulo para padronizar os dados
from sklearn.decomposition import PCA # modulo para aplicar PCA nos dados
from sklearn.preprocessing import MinMaxScaler

random.seed(1) # inicia a semente do gerador de números aleatórios. 
os.chdir(r'D:\git\MBA_DataScience\modulo_1\Introducao-Data-Science\data') # change directory to folder with input files

# CSV file
data = pd.read_csv(r"D:\git\MBA_DataScience\modulo_1\Introducao-Data-Science\data\iris-with-error.csv", header=(0))

# considera somente os atributos, ignorando a última coluna, que contem a classe
y = np.array(data[data.columns[0:-1]])
X = np.array(data[data.columns[0:data.shape[1]-1]])


# Graded assignments 

vehicle = pd.read_csv('Vehicle.csv')
X = np.array(vehicle[vehicle.columns[0:vehicle.shape[1]-1]])

# Como temos 14 variaveis, podemos reduzir a dimensionalidade para qualquer valor abaixo disso
# Portanto, criamos uma lista com todas possibilidades para analizarmos qual é o minimo
# dimensoes que podemos reduzir ainda sendo possivel explicar sua variancia
n_components_values = np.arange(1, len(vehicle.columns)) # would be len(vehicle.columns) + 1 if it wasn't for the label column

# Vamos padronizar os dados, de modo a evitar o efeito da escala dos atributos.
scaler = StandardScaler().fit(X)
vehicle_transformed = scaler.transform(X)

# instanciamos o PCA sem especificar o numero de componentes que desejamos
# em seguida ajustamos ao nosso conjunto de dados
pca = PCA().fit(vehicle_transformed)

# mostra a variância acumulada para todos os possiveis numeros de componentes
# Notamos que com apenas duas variaveis conseguimos explicar 95% da variancia dos dados
plt.figure(figsize=(8, 5))
plt.plot(n_components_values, np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.xlabel('number of components', fontsize=20)
plt.ylabel('cumulative explained variance', fontsize=20)
plt.xticks(color='k', size=16)
plt.yticks(color='k', size=16)
plt.grid(True)
plt.show()