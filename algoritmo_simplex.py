# ---------------------------------------------------------------------------------------------------

# Algoritmo Simplex

# ---------------------------------------------------------------------------------------------------

# Bibliotecas

# ---------------------------------------------------------------------------------------------------

# Importa Bibliotecas
import numpy as np 
import pandas as pd
import plotly.graph_objects as go
import itertools as it
from plotly.offline import iplot, init_notebook_mode

# ---------------------------------------------------------------------------------------------------

# Função Simplex

# ---------------------------------------------------------------------------------------------------

# Define função simplex
def simplex(A, b, c):

  # Declara dimensão de linhas das restrições
  m = A.shape[0]

  # Declara dimensão de colunas das restrições
  n = A.shape[1] - m

  # Declara indices da matriz para combinções 
  indices = list(range(m + n))

  # Declara combinações
  combs = np.array(list(it.combinations(indices, n)))

  # Declare variável lógica de permissão
  permitido = False

  # Laço para número de combinações 
  for i in range(len(combs)):
    
    # Declara indices da variável livre
    livre = combs[i,:]     
    
    # Declara diferença de indices para pivos não livres
    pivos = np.setdiff1d(indices, livre)    
    
    # Declara vertice
    vertice = np.zeros(m+n)
    
    # Atualiza vertice com a solução dos pivôs da iteração
    vertice[pivos] = np.matmul(np.linalg.inv(A[:, pivos]), b)
    
    # Verifica se todos as variáveis do canto são maiores que 0
    if np.all(vertice >= 0):
      
      # Altera valor de permitido para verdadeiro
      permitido = True
      
      # Interrompe laço
      break

  # Condição mensagem de erro
  if permitido == False:
    raise Exception("Não existe solução para esse problema")

  # Declara nmatriz de variáveis livres
  N = A[:,livre]

  # Declara matriz de pivôs
  B = A[:,pivos]

  # Declara vetor de custo variáveis livres
  c_n = c[livre]

  # Declara vetor de custo pivos
  c_b = c[pivos]

  # Calcula r
  r = c_n - np.matmul( np.matmul(c_b, np.linalg.inv(B)), N)

  # Define nomes das colunas do dataframe
  nomes = ["X" + str(nome + 1) for nome in  range( n ) ] + ["Maximo"]
  
  # Define dataframe resultados
  resultados = pd.DataFrame(columns = nomes, index = range(100)  )

  # Declara contador do dataframe
  contador = 0

  # Laço para verificar solução menor que 0
  while np.any(r < 0):
    
    # Verfifica mínimo de r
    entrada = np.argmin(r)
    
    # Declara a inversa de B
    B_inv = np.linalg.inv(B)
    
    # Declara vetor u
    u = N[:, entrada]

    # Calcula B_inv_u
    B_inv_u = np.matmul(B_inv, u)

    # Laço para alterar B_inv_u igual a zero para 0.01
    for idx in range(len(B_inv_u)):
      if B_inv_u[idx] == 0:
        B_inv_u[idx] = 0.01
    
    # Calcula z
    z = np.matmul(B_inv, b) / B_inv_u
    
    # Declara coluna de saída
    saida = np.argmin(z)
    
    # Declara variável temporaria para substituir pivôs
    temp = pivos[saida]
    
    # Altera pivos
    pivos[saida] = livre[entrada]
    
    # Alterna variáveis livres
    livre[entrada] = temp
    
    # Altera matriz N
    N = A[:,livre]
    
    # Alterna matriz de pivos
    B = A[:,pivos]
    
    # Alterna vetor função c variáveis livres
    c_n = c[livre]
    
    # Alterna função c de pivos
    c_b = c[pivos]
    
    # Calcula r
    r = c_n - np.matmul( np.matmul( c_b, np.linalg.inv(B) ), N)

    # Declara vertice vetor de zeros
    vertice = np.zeros(m+n)
  
    # Atualiza canto com a solução dos pivôs da iteração
    vertice[pivos] = np.matmul(np.linalg.inv(A[:,pivos]), b)

    # Popula Xs do dataframe de resultados
    for var in range( n ):
      resultados.iloc[contador, var] = vertice[var]

    # Popula máximo do dataframe de resultados
    resultados.loc[contador, "Maximo"] = np.matmul(c, vertice) * -1

    # Atualiza contador
    contador += 1

  # Retorna data frame de resutados das iterações
  return resultados.dropna()


# ---------------------------------------------------------------------------------------------------

# Exemplo

# ---------------------------------------------------------------------------------------------------

# Declara matriz de coeficientes das restrições lado esquerdo das desigualdades
A = np.array([[3,  2,   1,   0,   0],
              [1,  0,   0,   1,   0],
              [0,  1,   0,   0,   1]]) 

# Declara vetor de coeficientes das restrições lado direito das desigualdades
b = np.array([18, 4, 6]) 

# Declara função objetivo
c = np.array([-3, -5, 0, 0, 0])

# Executa função
sol = simplex(A, b, c)

# ---------------------------------------------------------------------------------------------------

# Gráfico de Convergência

# ---------------------------------------------------------------------------------------------------

# Define variaveis z, y, z
x = sol.X1.values
y = sol.X2.values
z = sol.Maximo.values

# Declara Minimos e Maximos
xm = np.min(x)
xM = np.max(x)
ym = np.min(y)
yM = np.max(y)
zm = np.min(z)
zM = np.max(z)

# Declara limite do politopo extesão da convergência da função
xx = np.linspace(0, 2, 200)
yy = np.linspace(0, 3, 200)
zz = np.linspace(zm, zM, 100)

# Declara limite do politopo semiplano x = 4        
X1 = np.tile(np.array(4), (100, 200))
Y1 = np.tile(yy, (100, 1))
Z1 = np.tile(zz, (200, 1) ).transpose()

# Declara limite do politopo semiplano y = 6
X2 = np.tile(xx, (100, 1))
Y2 = np.tile(np.array(6), (100, 200))
Z2 = np.tile(zz, (200, 1) ).transpose()

# Calcula limite do politopo semiplano 3x + 2y = 18 
xx3 = np.linspace(0, 4, 200)
yy3 = (18 - 3 * xx3) / 2
zz = np.linspace(zm, zM, 100)

# Declara mascaras para x e y tendo y =< 6
mascara = yy3 <= 6
xx3_mascara = np.ma.MaskedArray(xx3, ~mascara)
xx3 = np.ma.filled(xx3_mascara.astype(float), np.nan)
yy3_mascara = np.ma.MaskedArray(yy3, ~mascara)
yy3 = np.ma.filled(yy3_mascara.astype(float), np.nan)

# Declara semiplano 3x + 2y = 18
X3 = np.tile(xx3, (100, 1))
for i in range(X3.shape[0]): 
  X3[i, 100] = 2
Y3 = np.tile(yy3, (100, 1))
for i in range(Y3.shape[0]): 
  Y3[i, 100] = 6
Z3 = np.tile(zz, (200, 1) ).transpose()

# Declara limite do politopo para 0 = x  e y = 6
xx4 = np.linspace(0, 4, 200)
yy4 = np.linspace(0, 6, 200)

# Declara limite do politopo semiplano x = 0        
X4 = np.tile(np.array(0), (100, 200))
Y4 = np.tile(yy4, (100, 1))
Z4 = np.tile(zz, (200, 1) ).transpose()

# Declara limite do politopo semiplano y = 0
X5 = np.tile(xx4, (100, 1))
Y5 = np.tile(np.array(0), (100, 200))
Z5 = np.tile(zz, (200, 1) ).transpose()

# Declara figura
fig = go.Figure(
    data=[go.Scatter3d(x= x, y= y, z= z,
                     mode="markers",
                     marker=dict(color="green", size=10),
                     name = "z = 3x + 5y",
                     showlegend = True),
          go.Surface(x=X1,
                     y=Y1,
                     z=Z1, 
                     colorscale= [[0, "rgb(255, 0, 0)"],
                                  [1, "rgb(255, 0, 0)"]],
                     showscale=False,
                     opacity =0.4,
                     name = "x = 4"),
          go.Surface(x=X2,
                     y=Y2,
                     z=Z2, 
                     colorscale= [[0, "rgb(255, 0, 0)"],
                                  [1, "rgb(255, 0, 0)"]],
                     showscale=False,
                     opacity =0.4,
                     name = "y = 6"),
          go.Surface(x=X3,
                     y=Y3,
                     z=Z3, 
                     colorscale= [[0, "rgb(255, 0, 0)"],
                                  [1, "rgb(255, 0, 0)"]],
                     showscale=False,
                     opacity =0.4,
                     name = "3x + 2y = 18"),
          go.Surface(x=X4,
                     y=Y4,
                     z=Z4, 
                     colorscale= [[0, "rgb(255, 0, 0)"],
                                  [1, "rgb(255, 0, 0)"]],
                     showscale=False,
                     opacity =0.4,
                     name = "x = 0"),
          go.Surface(x=X5,
                     y=Y5,
                     z=Z5, 
                     colorscale= [[0, "rgb(255, 0, 0)"],
                                  [1, "rgb(255, 0, 0)"]],
                     showscale=False,
                     opacity =0.4,
                     name = "y = 0")])
    
# Atualiza aparência da figura
fig.update_layout(scene = dict(xaxis=dict(range=[0, 4], autorange=False),
                               yaxis=dict(range=[0, 6], autorange=False),
                               zaxis=dict(range=[min(z), max(z)], autorange=False)),
                  title="Convergência Máximo - Algoritmo Simplex",
                  width=1000, height=800)

# Declara frames da figura
frames = [go.Frame(data= [go.Scatter3d(x=x[:k+1], 
                                       y=y[:k+1],
                                       z=z[:k+1])],
                   traces= [0],
                   name=f'frame{k}')for k  in  range(len(x))]
                  
# Atualiza frames da figura                  
fig.update(frames=frames),

# Atualiza aparência da figura
fig.update_layout(updatemenus=[dict(type="buttons",
                                    buttons=[dict(label="Play",
                                                  method="animate",
                                                  args=[None,
                                                        dict(frame=dict(redraw=True,
                                                                        fromcurrent=True))])])], showlegend=True)

# Desenha figura
iplot(fig)

# ---------------------------------------------------------------------------------------------------

