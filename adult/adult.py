import numpy as np
import pandas as pd
from sklearn import linear_model

#>50K, <=50K. 
#0  age: continuous. 
#1  workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
#2  fnlwgt: continuous. 
#3  education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
#4  education-num: continuous. 
#5  marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
#6  occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
#7  relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
#8  race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
#9  sex: Female, Male. 
#10 capital-gain: continuous. 
#11 capital-loss: continuous. 
#12 hours-per-week: continuous. 
#13 native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands."""


data = np.loadtxt("adult.data", delimiter=",", dtype='str')

print(pd.value_counts(data[:,0]))

for i in range(data.shape[1] - 1):
	coluna = data[:,i]
	atributos = pd.value_counts(coluna)
	data[data[:,i]==' ?', i] = atributos.idxmax()


















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

	
