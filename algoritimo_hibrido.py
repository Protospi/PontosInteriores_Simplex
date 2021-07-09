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

# Função hibrido

# ----------------------------------------------------------------------------------------------------------

# Define função método hibrido
def hibrido(n, a):
  
  # Declara matriz A
  A = np.zeros(shape=(a, a))
  
  # Laço para popular matriz A
  for l in range(A.shape[0]):
    A[l,:]  = np.random.choice(np.arange(1, 6), size = A.shape[1])
        
  # Declara vetor resposta coeficiente
  b = np.random.randint(10, 100, size=(a))
  
  # Declara função objetivo
  c = np.random.randint(1, 10, size=(n)) * -1
  
  # Declara vetor aleatorio
  aleatorio = np.random.randint(100,150, size=(n))
  
  # Define limites das variáveis
  res = [(0, 1000) for superior in aleatorio]
  
  # Calcula intercessao de planos
  x_novo = np.linalg.solve(A, b)
  
  sol2 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex', x0 = x_novo)
  
  sol3 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex')

# ----------------------------------------------------------------------------------------------------------

# Simula algoritmo Hibrido

# ----------------------------------------------------------------------------------------------------------

# Declara variaveis
hibridos_i = np.zeros(shape=(100, 100))

# Laço para calcular máximos da função objetivo e número de iterações
for i in np.arange(2, hibridos_i.shape[0]):
  j = 1
  while(j <= i):
    hibridos_i[i, j] = hibrido(i, j)
    j += 1 


# ----------------------------------------------------------------------------------------------------------

# Gráficos de calor das interações Hibrido

# ----------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Declara array simplex simulações
df_hibrido = np.vstack( [hibridos_i.transpose(), np.zeros(100) ] )
df_hibrido[df_hibrido == 0] = np.nan

# Define heatmap
ax = sns.heatmap(df_hibrido,
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
ax.set_title("Simulação do Algoritmo Hibrido", 
             fontsize = 18,
             fontweight = "bold")
             
# Adiciona grids ao grafico e define limites
plt.grid()
plt.xlim(2,)
plt.ylim(1,100)
ax.set_facecolor("black")

# Desenha gráfico
plt.show()

# ----------------------------------------------------------------------------------------------------------

a = 3

n = 3

# Declara matriz A
A = np.zeros(shape=(a, a))

# Laço para popular matriz A
for l in range(A.shape[0]):
  A[l,:]  = np.random.choice(np.arange(1, 11), size = A.shape[1], replace = False)
      
# Declara vetor resposta coeficiente
b = np.random.randint(50, 100, size=(a))

# Declara função objetivo
c = np.random.randint(1, 10, size=(n)) * -1

# Declara vetor aleatorio
aleatorio = np.random.randint(10,50, size=(n))

# Declara vetor aleatorio
maximos = np.matmul(A, aleatorio )

# Define limites das variáveis
res = [(0, superior) for superior in aleatorio]

# Executa optimização linprog pontos iteriores
sol1 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='interior-point', options= {'tol' : 0.1} )

# Declara distancias
distancias = np.zeros(A.shape[0])
divisor = np.zeros(A.shape[0])

# Declara laço de distancias
for i in range(A.shape[0]):
  for j in range(A.shape[1]):
    distancias[i] += A[i,j] * sol1.x[j]
    divisor[i] += A[i,j] ** 2
    
  distancias[i] += b[i]
  distancias[i] = distancias[i] / np.sqrt(divisor[i]) 

# Recupera indices de distancias
ordem = np.argsort(distancias)

# Declara matriz de desigualdades
A1 = np.stack([A[ordem[0], :],
               A[ordem[1], :]])
               
# Declara b
b1 = np.array([b[ordem[0]], b[ordem[1]]])

# Calcula intercessao de planos
x_novo = np.linalg.solve(A1, b1) 

# Executa optimização linprog simplex
sol2 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex', x0 = x_novo)

sol3 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='simplex')

print("----------------------------------------------")

# Retorno da função Hibrido
print("F = ", sol2.fun)
print("It = ", sol2.nit +sol1.nit)


print("----------------------------------------------")


# Retorno da função Simplex
print("F = ", sol3.fun)
print("It = ", sol3.nit)


print("----------------------------------------------")





















a = 10

n = 10

# Declara matriz A
A = np.zeros(shape=(a, a))

# Laço para popular matriz A
for l in range(A.shape[0]):
  A[l,:]  = np.random.choice(np.arange(1, 6), size = A.shape[1])
      
# Declara vetor resposta coeficiente
b = np.random.randint(10, 100, size=(a))

# Declara função objetivo
c = np.random.randint(1, 10, size=(n)) * -1

# Declara vetor aleatorio
aleatorio = np.random.randint(100,150, size=(n))

# Define limites das variáveis
res = [(0, 1000) for superior in aleatorio]

# Calcula intercessao de planos
x_novo = np.linalg.solve(A, b)

sol2 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex', x0 = x_novo)

sol3 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex')


print("----------------------------------------------")

# Retorno da função Hibrido
print("F = ", sol2.fun)
print("It = ", sol2.nit)

print("----------------------------------------------")

# Retorno da função Simplex
print("F = ", sol3.fun)
print("It = ", sol3.nit)

print("----------------------------------------------")




# Declara contador
contador = 0

# Declara tamanho da matriz de condicionamento
a = 5

# Declara numero de dimensoes
n = 5


for i in range(100):
  
  # Declara matriz A
  A = np.zeros(shape=(a, a))
  
  # Laço para popular matriz A
  for l in range(A.shape[0]):
    A[l,:]  = np.random.choice(np.arange(1, 6), size = A.shape[1])
   
  # Declara vetor resposta coeficiente
  b = np.random.randint(10, 100, size=(a))
  
  # Declara função objetivo
  c = np.random.randint(1, 10, size=(n)) * -1
  
  # Define limites das variáveis
  res = [ (0, superior) for superior in np.repeat(10, n) ] 
  
  # Calcula intercessao de planos
  x_novo = np.linalg.solve(A, b)
  
  sol2 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex', x0 = x_novo)
  
  sol3 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex')
  
  if np.round(sol2.fun, 10) == np.round(sol3.fun, 10):
    contador += 1




























# Define Matriz A lado esquerdo das equações de desiguladade das restrições
A = np.array([[1,0],
              [0,1],
              [3, 2],
              [-1, 0],
              [0, -1]])

# Define função objetivo o
c = np.array([3,5]) * -1

# Define vetor lado direito das desigualdades das restrições
b = np.array([4, 6, 18, 0, 0])

# Define vetor X0
x0 = np.array([1, 1])

# Define limites das variáveis
res = [(0, superior) for superior in np.random.randint(1,50, size=(c.shape[0]))]



# Executa optimização linprog pontos iteriores
sol1 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='interior-point', options= {'tol' : 0.1} )

# Declara distancias
distancias = np.zeros(A.shape[0])

# Declara laço de distancias
for i in range(A.shape[0]):
  for j in range(A.shape[1]):
    distancias[i] += A[i,j] * sol1.x[j]
    divisor[i] += A[i,j] ** 2
    
  distancias[i] -= b[i]
  distancias[i] = distancias[i] / np.sqrt(divisor[i]) 

# Recupera indices de distancias
ordem = np.argsort(distancias)

# Declara matriz de desigualdades
A1 = np.stack([A[ordem[-1], :],
               A[ordem[-2], :]])
               
# Declara b
b1 = np.array([b[ordem[-1]], b[ordem[-2]]])

x_novo = np.linalg.solve(A1, b1)

# Executa optimização linprog simplex
sol2 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='revised simplex', x0 = x_novo)

sol3 = linprog(c, A_ub=A, b_ub=b, bounds=(res),  method='simplex')

# Retorno da função
print(sol2.fun, sol2.nit + sol1.nit, sol2.x)

print(sol3.fun, sol3.nit, sol3.x)

# ----------------------------------------------------------------------------------------------------------








# ----------------------------------------------------------------------------------------------------------

# Testes

# ----------------------------------------------------------------------------------------------------------

# Declara matriz de coeficientes das restrições lado esquerdo das desigualdades
A = np.array([[3,  2],
              [1,  0],
              [0,  1]]) 

# Declara vetor de coeficientes das restrições lado direito das desigualdades
b = np.array([18, 4, 6]) 

# Declara função objetivo
c = np.array([-3, -5])

# Resolve com Simplex com Ponto inicial
simplex = linprog(c, A_ub=A, b_ub=b,  method='revised simplex')

# # Ilumina com Simplex
pi = linprog(c, A_ub=A, b_ub=b,  method='interior-point', options= {"tol": 0.2})

# Limites inferiores de x
x_inf = np.ceil(pi.x)

# Define limites das variáveis
res = [(inferior, None) for inferior in x_inf]

# Reesolve com Simplex
hibrido = linprog(c, A_ub=A, bounds=res,  b_ub=b,  method='revised simplex')

# ----------------------------------------------------------------------------------------------------------

a = 3

n = 3

# Declara matriz A
A = np.zeros(shape=(a, n))

# Laço para popular matriz A
for l in range(A.shape[0]):
  A[l,:]  = np.random.choice(np.arange(1, 11), size = A.shape[1])
      
# Declara vetor resposta coeficiente
b = np.random.randint(0, 100, size=(a))

# Declara função objetivo
c = np.random.randint(1, 10, size=(n)) * -1

# Define limites das variáveis para simplex
res_1 = [(res, 10000) for res in np.zeros(n)]

# # Ilumina com Simplex
pi = linprog(c, A_ub=A, b_ub=b, bounds=res_1,  method='interior-point')

# Limite inferior de x
x_inferior = np.floor(pi.x)

# Resolve com Simplex com Ponto inicial
simplex = linprog(c, A_ub=A, b_ub=b, bounds=res_1,  method='simplex')

# Define limites das variáveis para hibrido
res_2 = [(inferior, 10000) for inferior in x_inferior]

# Reesolve com Simplex
hibrido = linprog(c, A_ub=A, bounds=res_2,  b_ub=b,  method='simplex')

# Imprime resultados
print("X = ", simplex.x)
print("Max = ", simplex.fun)
print("Iter = ", simplex.nit)
print("X = ", hibrido.x)
print("Max = ", hibrido.fun)
print("Iter = ", hibrido.nit)













a = 50

n = 50

# Declara matriz A
A = np.zeros(shape=(a, n))

# Laço para popular matriz A
for l in range(A.shape[0]):
  A[l,:]  = np.random.choice(np.arange(1, 11), size = A.shape[1])
      
# Declara vetor resposta coeficiente
b = np.random.randint(10, 100, size=(a))

# Declara função objetivo
c = np.random.randint(1, 10, size=(n)) 


# Define limites das variáveis para simplex
res_1 = [(res, 100) for res in np.zeros(n)]

linprog(c * -1, A_ub=A, b_ub=b, bounds=res_1,  method='simplex')


pi = pontos_interiores(A, b, c, np.repeat(0.0001, n), 0.999999, 0.2)


# Limite inferior de x
x_inferior = np.floor( pi[ pi.shape[0] - 2: pi.shape[0] - 1 ] )

# calcula maximo
max_arg = np.argmax(x_inferior.iloc[:,0:-1])


# Define limites das variáveis para hibrido
res_1[max_arg] = (int(x_inferior.iloc[:,max_arg].values), 100)


linprog(c * -1, A_ub=A, b_ub=b, bounds=res_1,  method='simplex')


