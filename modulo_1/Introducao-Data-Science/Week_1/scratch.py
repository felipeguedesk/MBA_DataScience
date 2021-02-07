from statistics import mode
import numpy as np 
import matplotlib.pyplot as plt #biblioteca gráfica para mostrar os gráficos
import pandas as pd # biblioteca pandas
from scipy.stats import entropy
from scipy.stats import iqr
import math
from scipy.stats import pearsonr, spearmanr

# # Exercicios de fixação:
# # 1)
# result_map = {}
# for p in np.arange(0, 1.01, 0.01):
#     result_map[p] = entropy([p, 1-p], base=2)

# plt.figure(num=1)
# plt.plot(result_map.keys(), result_map.values()) 
# plt.xlabel("Entropy",fontsize = 20) 
# plt.ylabel("Probability of heads", fontsize = 20)
# plt.show()
# var = np.var(p)

# # 2)
# poisson_results = {}
# for l in np.arange(0, 0.4, 0.1):
#     distribution = np.random.poisson(l, 500)
#     poisson_results[l] = [np.mean(distribution), np.var(distribution)]
# means = [meanVar[0] for meanVar in poisson_results.values() if isinstance(meanVar, list) and len(meanVar) > 1]
# variances = [meanVar[1] for meanVar in poisson_results.values() if isinstance(meanVar, list) and len(meanVar) > 1]
# plt.figure(num=2)
# plt.plot(poisson_results.keys(), means, color='blue', label='Mean') 
# plt.plot(poisson_results.keys(), variances, color='red', label='Var') 
# plt.xlabel("Poisson lambda",fontsize = 20) 
# plt.ylabel("Mean/Var", fontsize = 20)
# plt.legend()
# plt.show()

# Weekly Evaluation
# 1 
# Z = [0,0,2,3,1,2,4,5,6,8,0,0,7,0,1,2,1,1, 5,3,1,0,7,3,2,3,4,5,6,7,8,9,1,1,2,2,3]
# print("A moda é: {}".format(mode(Z)))

# # 2
# Z = [0,0,2,3,1,2,4,5,6,8,0,0,7,0,1,2,1,1, 5,3,1,0,7,3,2,3,4,5,6,7,8,9,1,1,2,2,3]

# print("The mean is {} and the median is {}".format(np.mean(Z), np.median(Z)))

# # 3
# Z = [0,0,2,3,1,2,4,5,6,8,7,5,3,1,0,7,3,2,3,4,5,6,7,8,9,1,1,2,2,3]
# def variancia(X):
#     m = np.mean(X)
#     N = len(X)
#     s = 0
#     for i in np.arange(0, len(X)):
#         s = s + (X[i]-m)**2
#     s = s/(N-1)
#     return s

# print("Var={},StdDev={},IQR={}".format(variancia(Z), math.sqrt(variancia(Z)), iqr(Z)))

# # 4

# x = np.linspace(-1.5, 1.5, num=100)
# tan = [math.tan(value) for value in x]
# corr, p_value = pearsonr(x, tan)
# corrs, p_values = spearmanr(x, tan)
# corr = int(corr*100)/100
# corrs = int(corrs*100)/100
# print("Pearson={}, Spearman={}".format(corr, corrs))

# 5
P = [0.15,0.25,0.2]
Q = [0.3, 0.6, 0.1]
print('KL(P,Q) = ', entropy(P,Q, base = np.exp(1)))
print('KL(Q,P) = ', entropy(Q,P, base = np.exp(1)))