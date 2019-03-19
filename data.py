import numpy as np
import pandas as pd

#carrega sem o cabeçalho, sem as duas primeirasa colunas e sem as 2 ultimas colunas
data = np.loadtxt("AirQualityUCI.csv", dtype=str, delimiter=';',skiprows=1)[:,2:-2] 

#exclui todas linhas vazias do final do arquivo
data = data[:np.nonzero(data=='')[0][0],:]

#converte toda a tabela em float substituindo virgulas por pontos.
data = np.char.replace(data,',','.').astype(np.float)

#salva 4ª coluna
y = data[:,3]

#deleta 4ª coluna do data
data = np.delete(data, 3, axis=1)


#remove colunas com excesso de valores invalidos
for i in range(data.shape[1]-1,-1,-1):
	if np.count_nonzero(data[:,i]==-200) > int(0.1*data.shape[0]):
		data = np.delete(data,i,axis=1)


#atualiza valores invalidos
for i in range(data.shape[1]):
	avg = np.mean(data[np.nonzero(data[:,i] != -200), i])
	data[np.nonzero(data[:,i] == -200), i] = avg

#salva o arquivo limpo
finalData = np.insert(data, data.shape[1], y, axis=1)
np.savetxt('AirQFinal.cvs', finalData, delimiter=';', fmt='%f')

