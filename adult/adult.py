import numpy as np
import pandas as pd
from sklearn import linear_model

data = np.loadtxt("adult.data", delimiter=",", dtype='str')

print(pd.value_counts(data[:,0]))

for i in range(data.shape[1] - 1):
	coluna = data[:,i]
	atributos = pd.value_counts(coluna)
	data[data[:,i]==' ?', i] = atributos.idxmax()

print(pd.value_counts(data[:,0]))



















#data = np.place(data[:,-1:], data[:,-1:] == y[0], 0)
#data = np.place(data[:,-1:], data[:,-1:] == y[1], 1)

#y = data[:,-1:]

#avg = np.mean(data[data[:,3]!='?',3].astype(np.float))

#data[data[:,3]=='?', 3] = avg	

#x = data[:,1:].astype(np.float)
#y = data[:,0].astype(np.float)


#tam = int(data.shape[0]*0.7)

#xtr = x[:tam, :]
#ytr = y[:tam]

#xte = x[tam:, :]
#yte = y[tam:]

#reg = linear_model.LinearRegression()
#reg.fit(xtr, ytr)

#y_pred = reg.predict(xte)

#print((abs(y_pred - yte).sum())/len(yte))

	
