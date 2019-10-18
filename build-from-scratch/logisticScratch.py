import numpy as np

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

print(X)

def GD(X,y,theta,eta = 0.05):
    thetaOld = theta
    thetaEpoch= theta
    N = X.shape[0]
    for it in range(10000):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:,i].reshape(X.shape[0],1)
            yi = y[i]
            hi = 1.0/(1+np.exp(-np.dot(thetaOld.T, xi)))

            gi = (yi-hi)*xi

            thetaNew = thetaOld + eta * gi
            thetaOld = thetaNew
            if(np.linalg.norm(thetaEpoch - thetaOld) < 1e-4):
                break
            thetaEpoch = thetaOld
    
    return thetaEpoch,it


def predicts(theta, x):
   return  1/(1 + np.exp(-theta[0]*x + theta[1]))



d = X.shape[0]

theta_init =  np.random.randn(d, 1)


theta,it = GD(X,y,theta_init)

print(theta)

predict = predicts(theta,5.50)

if(predict > 0.5):
    print("Tach")
else: 
    print("pass")