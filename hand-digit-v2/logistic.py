from sklearn.datasets import fetch_mldata 


mnist  = fetch_mldata('mnist-original',data_home='./')
x_all =mnist.data
y_all = mnist.target 


import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np

#Show some number in dataset 

plt.imshow(x_all.T[:,3000].reshape(28,28))
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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#chia du lieu thanh 2 tap train va test 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1000)

# train model 

model = LogisticRegression(C=1e5)
model.fit(x_train,y_train)

y_prediction = model.predict(x_test)

print("accuracy" + str(100* accuracy_score(y_test,y_prediction)))

#save model 

from sklearn.externals import joblib

joblib.dump(model,"digits.pkl",compress=3)