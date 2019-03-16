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

x = pd.DataFrame(data)
x = x.interpolate(data)
x = np.array(x)
