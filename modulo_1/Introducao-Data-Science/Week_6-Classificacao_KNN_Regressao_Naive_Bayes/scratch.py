import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.datasets as skdata
from mlxtend.plotting import plot_decision_regions

np.random.seed(42) # define the seed (important to reproduce the results)

data = pd.read_csv('../data/vertebralcolumn-2C.csv', header=(0))
data = data.dropna(axis='rows') #remove NaN

data = data.to_numpy()
nrow,ncol = data.shape
y = data[:,-1]
X = data[:,0:ncol-1]

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

p = 0.2 # fraction of elements in the test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = p, random_state = 42)

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('Accuracy in Naive Bayes classification: ', model.score(x_test, y_test))


# Gera os dados em duas dimensões

n_samples = 100 # número de observações

# centro dos grupos

centers = [(-4, 0), (0, 0), (3, 3)]

X, y = skdata.make_blobs(n_samples=100, n_features=2, cluster_std=1.0, centers=centers, 

                         shuffle=False, random_state=42)

plt.figure(figsize=(6,4))

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=50, alpha=0.7)

plt.show()
p = 0.2 # fraction of elements in the test set

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = p)
model = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('Acurácia: ', model.score(x_test, y_test))
plt.figure(figsize=(6,4))

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=50, alpha=0.7)

plot_decision_regions(X, y, clf=model, legend=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Regions for Logistic classification')
plt.show()

