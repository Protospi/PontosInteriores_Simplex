# ------------------------------------------------------------------------------------------------------------

# Algoritmo de Pontos Interiores

# ------------------------------------------------------------------------------------------------------------

# Bibliotecas

# ------------------------------------------------------------------------------------------------------------

# Importa Bibliotecas
import numpy as np 
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode

# ------------------------------------------------------------------------------------------------------------

# Função Pontos Interiores

# ------------------------------------------------------------------------------------------------------------

# Define função pontos_interiores
def pontos_interiores(A, b, c, x0, gamma, tol):
  
  # Define resposta Máximos
  respostas = np.zeros(51)
  
  # Define nomes
  nomes = ["X" + str(n + 1) for n in  range( A.shape[1] ) ] + ["Maximo"]
  
  # Define Resultados
  resultados = pd.DataFrame(columns = nomes, index = range(50)  )
  
  # Adiciona X0
  for k in range( A.shape[1] ):
    resultados.iloc[0, k] = x0[k]
  
  # Calcula função no ponto inicial x0
  resultados.loc[0, "Maximo"] = np.matmul( np.transpose(c), x0 )
  
  # Define vetor x0
  x = x0
  
  # Laço para calcular respostas e popular dataframe de respostas
  for i in range(resultados.shape[0]):
    
    # Define vetor Vk
    vk = b - np.matmul(A, x) 
    
    # Declara matriz Dk
    Dk = np.identity(  A.shape[0] )
    
    # Atualiza matriz Dk
    np.fill_diagonal(Dk, Dk.diagonal() / vk )
    
    # Define dx
    dx = np.matmul(np.linalg.inv( np.matmul(np.matmul(np.matmul(np.transpose(A), Dk), Dk), A) ), c)
    
    # Define dv
    dv = np.matmul(-A, dx)
    
    # Define passo
    passo = vk / dv
    
    # Laço para substituir positivo por -1000
    for j in range( len(passo) ):
      if passo[j] >= 0:
        passo[j] = -1000
  
    # Define alpha
    alpha = gamma * np.max(passo)
    
    # Define próximo x
    x = np.subtract(x, alpha * dx)
    
    # Calcula valor da função
    respostas[i+1] = np.matmul( np.transpose(c), x )
    
    # Popula resultados adicionando Xs
    for k in range( A.shape[1] ):
      resultados.iloc[i+1, k] = x[k]
    
    # Popula resultados adicionando valor da função
    resultados.loc[i+1, "Maximo"] =  respostas[i+1]
    
    # Condição de parada do algoritmo para incremento da resposta i+1 - i < tolerância
    if respostas[i+1] - respostas[i] < tol:
      break
  
  # Retorno da função
  return resultados.dropna()

# ------------------------------------------------------------------------------------------------------------

# Exemplo

# ------------------------------------------------------------------------------------------------------------

# Define Matriz A lado esquerdo das equações de desigualdade das restrições
A = np.array([[1,0],
              [0,1],
              [3, 2],
              [-1, 0],
              [0, -1]])

# Define função objetivo c
c = np.array([3,5])

# Define vetor lado direito das desigualdades das restrições
b = np.array([4, 6, 18, 0, 0])

# Define vetor X0
x0 = np.array([1, 1])

# Define gamma
gamma = 0.99

# Executa função
sol = pontos_interiores(A, b, c, x0, gamma, 0.1)

# ------------------------------------------------------------------------------------------------------------

# Gráfico de Convergência

# ------------------------------------------------------------------------------------------------------------

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

# Declara limite do politopo referente a resposta
xx = np.linspace(0, 2, 200)
yy = np.linspace(0, 3, 200)
zz = np.linspace(zm, zM, 100)

# Declara limite politopo semiplano x = 4        
X1 = np.tile(np.array(4), (100, 200))
Y1 = np.tile(yy, (100, 1))
Z1 = np.tile(zz, (200, 1) ).transpose()

# Declara limite politopo semiplano y = 6
X2 = np.tile(xx, (100, 1))
Y2 = np.tile(np.array(6), (100, 200))
Z2 = np.tile(zz, (200, 1) ).transpose()

# Calcula limite do politopo 3x + 2y = 18
xx3 = np.linspace(0, 4, 200)
yy3 = (18 - 3 * xx3) / 2
zz = np.linspace(zm, zM, 100)

# Mascara para y =< 6 limite do politopo 3x + 2y = 18
mascara = yy3 <= 6
xx3_mascara = np.ma.MaskedArray(xx3, ~mascara)
xx3 = np.ma.filled(xx3_mascara.astype(float), np.nan)
yy3_mascara = np.ma.MaskedArray(yy3, ~mascara)
yy3 = np.ma.filled(yy3_mascara.astype(float), np.nan)

# Declara limite politopo semiplano 3x + 2y = 18
X3 = np.tile(xx3, (100, 1))
for i in range(X3.shape[0]): 
  X3[i, 100] = 2
Y3 = np.tile(yy3, (100, 1))
for i in range(Y3.shape[0]): 
  Y3[i, 100] = 6
Z3 = np.tile(zz, (200, 1) ).transpose()

# Declara limite da função 0 <= x <= 4 e 0 <= y <= 6 
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
    data=[go.Scatter3d(x=[], y=[], z=[],
                     mode="lines",
                     line=dict(width=2, color="green"),
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
                  title="Convergência  Máximo - Algoritmo Pontos Interiores",
                  width=1000, height=800)

# Declara frames da figura
frames = [go.Frame(data= [go.Scatter3d(x=x[:k+1], 
                                       y=y[:k+1],
                                       z=z[:k+1])],
                   traces= [0],
                   name=f'frame{k}')for k  in  range(len(x)-1)]
                  
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

# ------------------------------------------------------------------------------------------------------------

