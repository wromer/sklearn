import numpy as np #Librería numérica
import matplotlib.pyplot as plt # Para crear gráficos con matplotlib
#%matplotlib inline # Si quieres hacer estos gráficos dentro de un jupyter notebook
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn



def f(x):  # función f(x) = 0.1*x + 1.25 + 0.2*Ruido_Gaussiano
    np.random.seed(42) # para poder reproducirlo
    y = 0.1*x + 1.25 + 0.2*np.random.randn(x.shape[0])
    return y
x = np.arange(0, 20, 0.5) # generamos valores x de 0 a 20 en intervalos de 0.5
y = f(x) # calculamos y a partir de la función que hemos generado
# hacemos un gráfico de los datos que hemos generado
plt.scatter(x,y,label='data', color='blue')
plt.title('Datos');

print("="*64)
print ("X")
print(x)

print("="*64)
print ("Y")
print(y)

'''
x=np.array([0, 1.8, 2.5, 3, 4.8, 5.2])
y=[2,3,5,3,1,2]
'''
x=np.array([4.8, 5.2])
y=[1,2]

print("="*64)
print ("X")
print(x)

print("="*64)
print ("Y")
print(y)


# Importamos la clase de Regresión Lineal de scikit-learn
from sklearn.linear_model import LinearRegression 
regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# instruimos a la regresión lineal que aprenda de los datos (x,y)
regresion_lineal.fit(x.reshape(-1,1), y) 
# vemos los parámetros que ha estimado la regresión lineal
print('w = ' + str(regresion_lineal.coef_) + ', b = ' + str(regresion_lineal.intercept_))
# resultado: w = [0.09183522], b = 1.2858792525736682


# vamos a predicir y = regresion_lineal(5)
nuevo_x = np.array([4.5]) 
print(nuevo_x)
print(nuevo_x.reshape(-1,1))

prediccion = regresion_lineal.predict(nuevo_x.reshape(-1,1))
print(prediccion)
# resultado: [1.7449]
'''
i=0
j=0
def w(i,j):  # función f(x) = 0.1*x + 1.25 + 0.2*Ruido_Gaussiano
    i=i+1
    print("i")
    print(i)
    print("j")
    print(j)
    i=i+1
    if i % 2==0:
        j=j+3
    else:
        if(i<9):
            w(i,j)
        else:
            return j


print(1%2)
print(2%2)
print(3%2)
print(4%2)
print(5%2)
print(6%2)
print(7%2)
    '''