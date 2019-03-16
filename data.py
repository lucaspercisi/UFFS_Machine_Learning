import numpy as np

#carrega sem o cabeçalho, sem as duas primeirasa colunas e sem as 2 ultimas colunas
data = np.loadtxt("AirQualityUCI.csv", dtype=str, delimiter=';',skiprows=1)[:,2:-2] 

#exclui todas linhas vazias do final do arquivo
data = data[:np.nonzero(data=='')[0][0],:]

print (data)
