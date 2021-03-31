from statistics import variance 
import math 
import numpy as np
import matplotlib.pyplot as plt
def linear_regression(x, y): 
    # número de observações/pontos
    n = np.size(x) 
  
    # médias de x e y
    m_x, m_y = np.mean(x), np.mean(y) 
    SS_xy = 0
    SS_xx = 0
    for i in range(0,len(x)):
        SS_xy = SS_xy + (x[i]-m_x)*(y[i]-m_y)
        SS_xx = SS_xx + (x[i]-m_x)**2
  
    # calcula os coeficientes de regressão
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return (b_0, b_1) 

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y = np.array([0, 3,4,5,10,8,12,15,15,19,22,21,26,28,27,29]) 
# estima os coeficientes
b = linear_regression(x, y) 
print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(b[0], b[1])) 

plt.plot(x, y, 'bo')
plt.xlabel("x", fontsize = 15)
plt.ylabel("y", fontsize = 15)
plt.show() 


def R2(x,y,b):
    n = len(y)
    c1 = 0
    c2 = 0
    ym = np.mean(y)
    for i in range(0,n):
        y_pred = b[0]+ x[i]*b[1] # valor predito
        c1 = c1 + (y[i]-y_pred)**2
        c2 = c2 + (y[i]-ym)**2
    R2 = 1 - c1/c2
    return R2

print('R2:', R2(x,y,b))