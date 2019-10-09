

def f(x) : 
    return x**2/(x-1)

def grad(x):
    return (x**2 - 2*x)/((x-1)**2)
# generate random x 

x_old = 3 
eta = 0.1 

for i in range(100): 
    x_new = x_old - eta * grad(x_old)
    if(abs(grad(x_new)) < 1e-3) :
        break
    x_old = x_new

print("nghiem x de f(x) nho nhat la =",x_new)