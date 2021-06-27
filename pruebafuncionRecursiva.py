
def w(i,j):  # función f(x) = 0.1*x + 1.25 + 0.2*Ruido_Gaussiano
    i=i+1
    print("i")
    print(i)
    print("j")
    print(j)
 
    if i % 2==0:
        j=j+3 
    if(i<7):
        return w(i,j)
    else:
        return j


result=w(0,0)
print(result)