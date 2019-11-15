## Doc lai du global_features va labels tu file da luu
## Chay mo hinh SVC


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import mahotas

features = np.loadtxt("feature-datas.txt")
labels = np.loadtxt("label-datas.txt")


x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.3)



## Train Mo hinh
C = 1.0
model = svm.SVC(kernel='linear', C=C)
model.fit(features, labels)
y_pred = model.predict(x_test)

print("Acc "  + str( accuracy_score(y_test,y_pred)));



# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier

# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# model = RandomForestClassifier(max_depth=5, n_estimators=10)
# model.fit(x_train, y_train)

# y_pred=model.predict(x_test)
# print("Accuracy: "+ str(100*accuracy_score(y_test,y_pred)))
