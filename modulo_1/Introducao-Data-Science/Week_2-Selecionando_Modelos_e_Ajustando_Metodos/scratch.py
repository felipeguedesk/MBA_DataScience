import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import random
# Pandas is used for data manipulation
import pandas as pd
import matplotlib.pyplot as plt

random.seed(1)

#função para gerar os dados
def sine_function(x):
    return np.sin(2 * np.pi * x)

def x6_function(x):
    return x**6

#Funcao para calcular o erro quadrático médio
def rmse(a, b):
       return np.sqrt(np.mean(np.square(a - b)))

# # 1)
# rmsTestErrorPerN = {}
# rmsTrainErrorPerN = {}
# # training set
# for N_train in range(10, 1001, 5):
#     sigma = 0.1
#     x_train= np.linspace(0, 1,N_train)
#     y_train = sine_function(x_train) + np.random.normal(0,sigma, N_train)
#     x_train = x_train.reshape(len(x_train), 1)

#     # test set
#     N_test = 200
#     x_test=np.linspace(0, 1,N_test)
#     y_test = sine_function
# (x_test) +  np.random.normal(0,sigma, N_test)
#     x_test = x_test.reshape(-1, 1)

#     # Define a matriz de atributos
#     # Exercicios de fixação:
#     poly9 = PolynomialFeatures(degree=9)
#     x_train_pf = poly9.fit_transform(x_train)
#     model = linear_model.LinearRegression()
#     model.fit(x_train_pf,y_train)
#     x_test=np.linspace(0,1,200)
#     x_test=x_test.reshape(-1, 1)
#     x_test_pf=poly9.fit_transform(x_test)
#     y_test_pred = model.predict(x_test_pf)
#     y_train_pred = model.predict(x_train_pf)
#     errorTest = rmse(y_test_pred,y_test)
#     errorTrain = rmse(y_train_pred,y_train)
#     rmsTestErrorPerN[N_train] = errorTest
#     rmsTrainErrorPerN[N_train] = errorTrain
    
# sns.set() #apenas para deixar o plot mais "bonitinho" com fundo cinza e com grid
# plt.plot(rmsTestErrorPerN.keys(), rmsTestErrorPerN.values(), 'g-', label='RMS Test Error per N')
# plt.plot(rmsTrainErrorPerN.keys(), rmsTrainErrorPerN.values(), 'r-', label='RMS Training Error per N')
# plt.legend(fontsize=15)
# plt.show()


# # # 2)
# rmsTestErrorPerN = {}
# rmsTrainErrorPerN = {}

# # numero de amostras para o conjunto de treino
# n_train = 10
# n_test = 10

# # gera o conjunto de treino com valores no intervalo 0 e 1
# x_train = np.linspace(0, 1, n_train)
# y_train = x6_function(x_train) + np.random.normal(0, 0.1, n_train)

# # O conjunto de teste é criado da mesma forma que os conjunto de treino
# x_test=np.linspace(0,1,n_test)
# y_test = x6_function(x_test) +  np.random.normal(0, 0.1, n_test)

# for m in range(1, 10):
#     # Define a matriz de atributos
#     # Exercicios de fixação:
#     poly = PolynomialFeatures(degree=m)
#     x_train_pf = poly.fit_transform(x_train.reshape(-1, 1))
#     x_test_pf=poly.fit_transform(x_test.reshape(-1, 1))

#     model = linear_model.LinearRegression()
#     model.fit(x_train_pf,y_train)

#     y_test_pred = model.predict(x_test_pf)
#     y_train_pred = model.predict(x_train_pf)

#     errorTest = rmse(y_test_pred,y_test)
#     errorTrain = rmse(y_train_pred,y_train)
#     rmsTestErrorPerN[m] = errorTest
#     rmsTrainErrorPerN[m] = errorTrain

# plt.xlabel("Degree", fontsize = 15)
# plt.ylabel("RMSE", fontsize = 15)
# plt.plot(rmsTestErrorPerN.keys(), rmsTestErrorPerN.values(), 'r-', label='RMS Test Error per M')
# plt.plot(rmsTrainErrorPerN.keys(), rmsTrainErrorPerN.values(), 'b-', label='RMS Training Error per M')
# plt.legend(fontsize=15)
# plt.show()

# # # 3

# data = pd.read_csv("D:\git\MBA_DataScience\modulo_1\Introducao-Data-Science\Week_2\data\Vehicle.csv", header=(0)) # if parameter header is int represents the index of the row to be used as header
# data = data.dropna(axis='rows').values # remove NaN
# rowNum, columnNum = data.shape # shape returns size of data
# y = data[:,-1] # outcome/observation is the last row
# x = data[:,0:columnNum - 1] # remaining of table is input

# p = 0.2 # fracao de elementos no conjunto de teste

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = p)
# k = 3
# vk = []
# vscore = []
# for nkf in range(2, 30):
#     model = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
#     cv = cross_validate(model, x_train, y_train, cv=nkf)
#     print('folds:', nkf, 'Accuracy:', cv['test_score'].mean())
#     vscore.append(cv['test_score'].mean())
#     vk.append(nkf)

# best_k = np.argmax(vscore)+1
# print('Melhor k:', best_k)
# plt.figure(figsize=(10,5))
# plt.plot(vk, vscore, '-bo')
# plt.xlabel('k', fontsize = 15)
# plt.ylabel('Accuracy', fontsize = 15)
# plt.show()

# # 4

# data = pd.read_csv("D:\git\MBA_DataScience\modulo_1\Introducao-Data-Science\Week_2\data\iris.csv", header=(0)) # if parameter header is int represents the index of the row to be used as header
# data = data.dropna(axis='rows').values # remove NaN
# rowNum, columnNum = data.shape # shape returns size of data
# y = data[:,-1] # outcome/observation is the last row
# x = data[:,0:columnNum - 1] # remaining of table is input

# p = 0.2 # fracao de elementos no conjunto de teste

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = p)
# k = 3
# vk = []
# vscore = []
# for nkf in range(2, 20):
#     model = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
#     cv = cross_validate(model, x_train, y_train, scoring='accuracy', cv=nkf)
#     print('folds:', nkf, 'Accuracy:', cv['test_score'].mean())
#     vscore.append(cv['test_score'].mean())
#     vk.append(nkf)

# best_k = np.argmax(vscore)+1
# print('Melhor k:', best_k)
# plt.figure(figsize=(10,5))
# plt.plot(vk, vscore, '-bo')
# plt.xlabel('n_folds', fontsize = 15)
# plt.ylabel('Accuracy', fontsize = 15)
# plt.show()

# Assignment
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(10)
#função para gerar os dados
def function(x):
    y = x**2 + x**9
    return y

# training set
N_train = 10
sigma = 0.2
x_train= np.linspace(0, 1,N_train)
y_train = function(x_train) + np.random.normal(0,sigma, N_train)
x_train = x_train.reshape(len(x_train), 1)

# test set
N_test = 10
x_test=np.linspace(0, 1,N_test)
y_test = function(x_test) +  np.random.normal(0,sigma, N_test)
x_test = x_test.reshape(len(x_test), 1)
rmsTestErrorPerN = {}
rmsTrainErrorPerN = {}
errorCombinedPerN = {}
for m in range(1, 10):
    # Define a matriz de atributos
    # Exercicios de fixação:
    poly = PolynomialFeatures(degree=m)
    x_train_pf = poly.fit_transform(x_train)
    x_test_pf=poly.fit_transform(x_test)

    model = linear_model.LinearRegression()
    model.fit(x_train_pf,y_train)

    y_test_pred = model.predict(x_test_pf)
    y_train_pred = model.predict(x_train_pf)

    errorTest = rmse(y_test_pred,y_test)
    errorTrain = rmse(y_train_pred,y_train)
    rmsTestErrorPerN[m] = errorTest
    rmsTrainErrorPerN[m] = errorTrain
    errorCombinedPerN[m] = errorTrain + errorTest

print("The min combined error is: {}".format(min(errorCombinedPerN.values())))
plt.xlabel("Degree", fontsize = 15)
plt.ylabel("RMSE", fontsize = 15)
plt.plot(rmsTestErrorPerN.keys(), rmsTestErrorPerN.values(), 'r-', label='RMS Test Error per M')
plt.plot(rmsTrainErrorPerN.keys(), rmsTrainErrorPerN.values(), 'b-', label='RMS Training Error per M')
plt.plot(errorCombinedPerN.keys(), errorCombinedPerN.values(), 'g-', label='RMS Combined Error per M')
plt.legend(fontsize=15)
plt.show()