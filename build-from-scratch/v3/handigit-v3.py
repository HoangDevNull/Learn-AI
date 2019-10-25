
from sklearn.datasets import fetch_mldata 


mnist  = fetch_mldata('mnist-original',data_home='./')
x_all =mnist.data
y_all = mnist.target 


import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np

#Show some number in dataset 

# plt.imshow(x_all.T[:,3000].reshape(28,28))
# plt.axis("off")
# plt.show()


#loc lai chi con 2 chu so 0 va 1 

x0 = x_all[np.where(y_all == 0 )[0]]
x1 = x_all[np.where(y_all == 1)[0]]

y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])

# gop 0 va 1 lai thanh dataset 

x = np.concatenate((x0,x1), axis=0)
y = np.concatenate((y0,y1))

one = np.ones((x.shape[0],1))

x = np.concatenate((x,one), axis = 1)

print(x)
def GD(X,y,theta,eta = 0.05):
    thetaOld = theta
    thetaEpoch= theta
    N = X.shape[0]
    for it in range(10000):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i,:]
            yi = y[i]
            hi = 1.0/(1.0+np.exp(-np.dot(xi,thetaOld.T)))

            gi = (yi-hi)*xi

            thetaNew = thetaOld + eta * gi
            thetaOld = thetaNew
            if(np.linalg.norm(thetaEpoch - thetaOld) < 1e-3):
                break
            thetaEpoch = thetaOld
    
    return thetaEpoch,it

d = x.shape[1]

theta_init =  np.random.randn( 1,d)
print(theta_init)
theta,it = GD(x,y,theta_init)

np.savetxt('theta.txt',theta);






# train model 

# model = LogisticRegression(C=1e5)
# model.fit(x_train,y_train)

# y_prediction = model.predict(x_test)

# print("accuracy" + str(100* accuracy_score(y_test,y_prediction)))

# #save model 

# from sklearn.externals import joblib

# joblib.dump(model,"digits.pkl",compress=3)