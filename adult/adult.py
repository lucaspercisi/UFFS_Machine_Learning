from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

##########################################################


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

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

# substitui todos '?' pela atributo de maior incidência na coluna com excessão da ultima coluna
for i in range(data.shape[1] - 1):
	coluna = data[:,i]
	atributos = pd.value_counts(coluna)
	data[data[:,i]==' ?', i] = atributos.idxmax()

print(data[0,0], data[0,0].isdigit(), type(data[0,0]))
print(data[0,3], data[0,3].isdigit(), type(data[0,3]))

for i in range(data.shape[1]-1):
	coluna = np.unique(data[:, i])
	if not coluna[0].isdigit():
		for j in range(coluna.shape[0]):
			data[data[:,i] == coluna[j], i] = j

# workclass = np.unique(data[:, 1])
# for i in range(workclass.shape[0]):
# 	data[data[:, 1] == workclass[i], 1] = i
# #
# education = np.unique(data[:, 3])
# for i in range(education.shape[0]):
# 	data[data[:, 3] == education[i], 3] = i
#
# marital = np.unique(data[:, 5])
# for i in range(marital.shape[0]):
# 	data[data[:, 5] == marital[i], 5] = i
#
# occupation = np.unique(data[:, 6])
# for i in range(occupation.shape[0]):
# 	data[data[:, 6] == occupation[i], 6] = i
#
# relationship = np.unique(data[:, 7])
# for i in range(relationship.shape[0]):
# 	data[data[:, 7] == relationship[i], 7] = i
#
# race = np.unique(data[:, 8])
# for i in range(race.shape[0]):
# 	data[data[:, 8] == race[i], 8] = i
#
# sex = np.unique(data[:, 9])
# for i in range(sex.shape[0]):
# 	data[data[:, 9] == sex[i], 9] = i
#
# native = np.unique(data[:, 13])
# for i in range(native.shape[0]):
# 	data[data[:, 13] == native[i], 13] = i


Y = data[:,-1] #separa os rotulos
X = data[:,:-1] #remove os rotulos dos dados

# mapeia os rotulos
u = np.unique(Y)
np.place(Y, Y == u[0], 1)
np.place(Y, Y == u[1], 2)

train_data = int(data.shape[0]*0.7)

#separa e converte dados de treino
xtr = X[:train_data,:].astype(np.int)
ytr = Y[:train_data].astype(np.int)

#separa e converte dados de teste
xte = X[train_data:,:].astype(np.int)
yte = Y[train_data:].astype(np.int)

#Constroi a regressao logistica com os dados de trieno
model = linear_model.LogisticRegressionCV(max_iter=10000, refit=False, n_jobs=-1,
									  solver='liblinear', random_state=1008, penalty='l1').fit(xtr, ytr)

#faz a previsão com os dados de teste
y_hat = model.predict(xte)

#informe o erro com os rotulos separados.
print('Taxa de acerto (teste):', np.mean(y_hat == yte))

###########################################################################
###########################################################################
# model = linear_model.LogisticRegressionCV()
# parameters = {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
# 			  'fit_intercept':[True,False], 'max_iter':[1000, 10000, 100], 'refit':[True,False],
# 			  'intercept_scaling':[1,10,100,1000], 'multi_class':['ovr','auto'],'n_jobs':[-1]}
# grid = GridSearchCV(model,parameters, cv=None)
#
# grid.fit(xtr, ytr)
#
# print ("Melhor resultado: ", grid.best_score_)
# print("Residual sum of squares: %.2f" % np.mean((grid.predict(X_test) - y_test) ** 2))