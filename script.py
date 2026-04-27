import numpy as np

def activate(x):
    return 1 if x>=0 else -1

def read_data(n,m):
    x_data =np.zeros((n,m),dtype=float)
    t_data = np.zeros(n,dtype=int)
    print("Enter data")
    for i in range(n):
        row = np.array(list(map(float,input("Row " + str(i+1) + " : ").split())))
        x_data[i] = row[:m]
        label = int(row[m])
        t_data[i] = -1 if label==0 else 1
    
    return x_data,t_data

n = int(input("Number of samples: "))
m = int(input("Number of features: "))
x_data,t_data = read_data(n,m)
lr = float(input("Learning rate: "))
epochs = int(input("Enter epochs: "))

w = np.zeros(m,float)
b = 0.0

for _ in range(epochs):
    errors = 0
    for i in range(n):
        y = activate(np.dot(w,x_data[i]) + b)
        if y!=t_data[i]:
            w += lr * x_data[i] * t_data[i]
            b += lr * t_data[i]
            errors +=1

    
    print("Epoch ",_,"errors",errors,"w = ",w," b = ",b)
    if errors == 0:
        break


print("Final w = ",w)
print("Final b = ",b)


while True : 
    s = input("Test Sample or q")
    if s == 'q':
        break
    x  =np.array(list(map(float,s.split())))
    if len(x) !=m : 
        print("Invalid")
        continue
    
    y = activate(np.dot(w,x)+b)
    print("Predicted class",1 if y==1 else 0)
    


