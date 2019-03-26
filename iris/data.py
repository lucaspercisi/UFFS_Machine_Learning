#https://archive.ics.uci.edu/ml/index.php

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.linear_model import LogisticRegression
#from sklearn.datasets import load_iris

#carrega sem o cabeçalho, sem as duas primeirasa colunas e sem as 2 ultimas colunas
data = np.loadtxt("iris_data.txt", dtype=str, delimiter=',') 

data_origin = data

data = np.where(data=='Iris-setosa',0, data)
data = np.where(data=='Iris-virginica',1, data)
data = np.where(data=='Iris-versicolor',1, data)

data = data.astype(np.float)

np.random.seed(137)
np.random.shuffle(data)


x = data[:,:4]
y = data[:,-1]


training_x = x[:int(x.shape[0]*0.75),:]
training_y = y[:int(y.shape[0]*0.75)]
test_x = x[int(x.shape[0]*0.75):,:]
test_y = y[int(y.shape[0]*0.75):]

regression = LogisticRegression(random_state=0, solver='lbfgs', \
				multi_class='multinomial').fit(training_x, training_y)


pred_tr = regression.predict(training_x)

print('   Taxa de acerto (treino):', np.mean(pred_tr==training_y))

pred_te = regression.predict(test_x)

print('   Taxa de acerto (teste):', np.mean(pred_te==test_y))
