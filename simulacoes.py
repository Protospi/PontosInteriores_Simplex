# ----------------------------------------------------------------------------------------------------------

# Simulações

# ----------------------------------------------------------------------------------------------------------

# Define caminho conda no sistema
Sys.setenv(RETICULATE_PYTHON = "C:\\Users\\Drope\\miniconda3\\python.exe")

# ----------------------------------------------------------------------------------------------------------

# Importa pacotes
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------

# Declara função
def optimizar(n, a):

  # Declara matriz de restrições
  A = np.random.randint(100, size=(a * n)).reshape((a, n))

  # Declara vetor resposta coeficiente
  b = np.random.randint(10, 100, size=(a))

  # Declara função objetivo
  c = np.random.randint(1, 100, size=(n)) * -1

  # Define limites das variáveis
  res = [(0, superior) for superior in np.random.randint(1,50, size=(n))] 

  # Executa optimização linprog simplex e pontos interiores
  sol1 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='simplex')
  sol2 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='interior-point')

  # Retorno da função
  return(sol1.nit, -1 * sol1.fun, sol2.nit, -1 * sol2.fun)

# ----------------------------------------------------------------------------------------------------------

# Declara variaveis
pontos_interiores_i = np.zeros(shape=(100, 100))
simplex_i = np.zeros(shape = (100, 100))
pontos_interiores_f = np.zeros(shape=(100, 100))
simplex_f = np.zeros(shape = (100, 100))

# Laço para calcular máximos da função objetivo e número de iterações
for i in np.arange(2, pontos_interiores_i.shape[0]):
  j = 1
  while(j <= i):
    simplex_i[i, j], simplex_f[i, j], pontos_interiores_i[i, j], pontos_interiores_f[i, j] = optimizar(i, j)
    j += 1 

# ----------------------------------------------------------------------------------------------------------

# Graficos de calor das interações

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_simplex = np.vstack( [simplex_i.transpose(), np.zeros(100) ] )
df_simplex[df_simplex == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_simplex,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.50, 
                           "aspect": 50,
                           "label": "Iterações"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticks(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Variáveis', 
       ylabel= 'Restrições')
    
# Define titulo do gráfico       
ax.set_title("Simulação do Algoritmo Simplex", 
             fontsize = 18,
             fontweight = "bold")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3)

# Declara array simplex simulações
df_pi = np.vstack( [np.zeros(100),
                    pontos_interiores_i.transpose()[2, :],
                    pontos_interiores_i.transpose()[2:, :],
                    np.zeros(100) ] )
df_pi[df_pi == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_pi,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.50, 
                           "aspect": 50,
                           "label": "Iterações"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticks(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Variáveis', 
       ylabel= 'Restrições')
    
# Define titulo do gráfico       
ax.set_title("Simulação do Algoritmo Pontos Interiores", 
             fontsize = 18,
             fontweight = "bold")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(2,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

# Gráficos de Diferença  Pontos Interiores - Simplex 

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3)

# Declara array simplex simulações
df_dif = np.abs(simplex_f.transpose()[1:, :] - pontos_interiores_f.transpose()[1:, :])
df_dif[df_dif == 0] = np.nan
df_diferenca = np.round(np.log10(df_dif), 4) 

# Define heatmap
ax = sns.heatmap(df_diferenca,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.50, 
                           "aspect": 50,
                           "label": "Diferença (log 10)"})

# Altera rótulos dos eixos
ax.set(xlabel= 'Variáveis', 
       ylabel= 'Restrições')
       
# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticks(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.invert_yaxis()
    
# Define titulo do gráfico       
ax.set_title("Diferença de Máximos", 
             fontsize = 18,
             fontweight = "bold")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------
