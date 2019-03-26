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
pred_te = regression.predict(test_x)



#exclui todas linhas vazias do final do arquivo
#data = data[:np.nonzero(data=='')[0][0],:]

#converte toda a tabela em float substituindo virgulas por pontos.
#data = np.char.replace(data,',','.').astype(np.float)

#salva 4ª coluna
#y = data[:,3]

#deleta 4ª coluna do data
#data = np.delete(data, 3, axis=1)


#remove colunas com excesso de valores invalidos
#for i in range(data.shape[1]-1,-1,-1):
#	if np.count_nonzero(data[:,i]==-200) > int(0.1*data.shape[0]):
#		data = np.delete(data,i,axis=1)


#atualiza valores invalidos
#for i in range(data.shape[1]):
#	avg = np.mean(data[np.nonzero(data[:,i] != -200), i])
#	data[np.nonzero(data[:,i] == -200), i] = avg

#salva o arquivo limpo
#finalData = np.insert(data, data.shape[1], y, axis=1)
#np.savetxt('AirQFinal.cvs', finalData, delimiter=';', fmt='%f')

