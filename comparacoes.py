# ----------------------------------------------------------------------------------------------------------

# Simulações para comparar Pontos Interiores, Simplex e Simplex revisado

# ----------------------------------------------------------------------------------------------------------

# Importa pacotes
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------------------------------------------------------------

# Declara função de Optimização

# ----------------------------------------------------------------------------------------------------------

# Declara função
def optimizar(V, R):

  # Declara matriz de restrições
  A = np.random.randint(1000, size=(R * V)).reshape((R, V))

  # Declara vetor resposta coeficiente
  b = np.random.randint(1000, 10000, size=(R))

  # Declara função objetivo
  c = np.random.randint(1, 1000, size=(V)) * -1

  # Define limites das variáveis
  limites_x = [(0, superior) for superior in np.random.randint(1, 100, size=(V))] 

  # Executa optimização linprog simplex 
  inicio_s = time.time()
  s = linprog(c, A_ub=A, b_ub=b, bounds=(limites_x),  method='simplex')
  fim_s = time.time()
  s_t = fim_s - inicio_s
  
  # Executa optimização linprog simplex revisado
  inicio_rs = time.time()
  rs = linprog(c, A_ub=A, b_ub=b, bounds=(limites_x),  method='revised simplex')
  fim_rs = time.time()
  rs_t = fim_rs - inicio_rs
  
  # Executa optimização linprog pontos interiores
  inicio_pi = time.time()
  pi = linprog(c, A_ub=A, b_ub=b, bounds=(limites_x),  method='interior-point')
  fim_pi = time.time()
  pi_t = fim_pi - inicio_pi

  # Retorno da função
  return(s.nit, -1 * s.fun, s_t, pi.nit, -1 * pi.fun, pi_t, rs.nit, rs.fun, rs_t)

# ----------------------------------------------------------------------------------------------------------

# Declara função de Simulação

# ----------------------------------------------------------------------------------------------------------

# Tamanhodas simulacao
V = 100

# Declara variaveis
pi_i = np.zeros(shape=(V, V))
pi_f = np.zeros(shape=(V, V))
pi_c = np.zeros(shape=(V, V))
s_i = np.zeros(shape = (V, V))
s_f = np.zeros(shape = (V, V))
s_c = np.zeros(shape = (V, V))
rs_i = np.zeros(shape = (V, V))
rs_f = np.zeros(shape = (V, V))
rs_c = np.zeros(shape = (V, V))

for v in np.arange(2, V):
  r = 1
  while(r <= v):
    s_i[v, r], s_f[v, r], s_c[v, r], pi_i[v, r], pi_f[v, r], pi_c[v, r], rs_i[v, r], rs_f[v, r], rs_c[v, r] = optimizar(v, r)
    r += 1

# ----------------------------------------------------------------------------------------------------------

# Graficos mapa de calor interações Simplex

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_simplex = s_i.transpose()
df_simplex[df_simplex == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_simplex,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
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
ax.set_title("Número de Iterações Simplex", 
             fontsize = 18,
             fontweight = "bold")
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()


# ----------------------------------------------------------------------------------------------------------

# Gráficos mapa de calor iterações Simplex Revisado 

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_rs = rs_i.transpose()
df_rs[df_rs == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_rs,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
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
ax.set_title("Número de Iterações Simplex Revisado", 
             fontsize = 18,
             fontweight = "bold")
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

# Gráficos mapa de calor iterações Pontos Interiores

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3)

# Declara array simplex simulações
df_pi =  pi_i.transpose()
df_pi[df_pi == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_pi,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
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
ax.set_title("Número de Iterações Pontos Interiores", 
             fontsize = 18,
             fontweight = "bold")
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(2,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

# Gráficos mapa de calor de diferença Pontos Interiores - Simplex 

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3)

# Declara array simplex simulações
df_dif_pi_s = np.abs(s_f.transpose() - pi_f.transpose())
df_dif_pi_s[df_dif_pi_s == 0] = np.nan
df_dif_pi_s = np.round(np.log10(df_dif_pi_s), 4) 

# Define heatmap
ax = sns.heatmap(df_dif_pi_s,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
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
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()


# ----------------------------------------------------------------------------------------------------------

# Gráficos mapa de calor de diferença Pontos Interiores - Simplex Revisado

# ----------------------------------------------------------------------------------------------------------


# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3)

# Declara array simplex simulações
df_dif_pi_rs = np.abs(rs_f.transpose() - pi_f.transpose())
df_dif_pi_rs[df_dif_pi_rs == 0] = np.nan
df_dif_pi_rs = np.round(np.log10(df_dif_pi_rs), 6) 

# Define heatmap
ax = sns.heatmap(df_dif_pi_rs,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
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
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

# Gráfico mapa de calor custo computacional Simplex

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_custo_s = s_c.transpose()
df_custo_s[df_custo_s == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_custo_s,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Tempo(s)"})

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
ax.set_title("Custo Computacional Simplex", 
             fontsize = 18,
             fontweight = "bold")
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()



# ----------------------------------------------------------------------------------------------------------

# Gráfico mapa de calor custo computacional Simplex Revisado

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_custo_rs = rs_c.transpose()
df_custo_rs[df_custo_rs== 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_custo_rs,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Tempo(s)"})

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
ax.set_title("Custo Computacional Simplex Revisado", 
             fontsize = 18,
             fontweight = "bold")
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

# Gráfico mapa de calor custo computacional Pontos interiores

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_cuto_pi = pi_c.transpose()


# Define heatmap
ax = sns.heatmap(df_cuto_pi,
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Tempo(s)"})

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
ax.set_title("Custo Computacional Pontos Interiores", 
             fontsize = 18,
             fontweight = "bold")
             
# Define cor de fundo do grafico
ax.set_facecolor("black")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)

# Desenha gráfico
plt.show()


# ----------------------------------------------------------------------------------------------------------

