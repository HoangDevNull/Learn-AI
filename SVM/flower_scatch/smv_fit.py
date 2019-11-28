from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import mahotas


X = np.loadtxt("feature-datas.txt")
y = np.loadtxt("label-datas.txt")

params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# svm_model = GridSearchCV(SVC(), params_grid, cv=5)
# svm_model.fit(X, y)


model = SVC(kernel = 'linear', C = 10)
model.fit(X, y)

w1 = model.coef_
b1 = model.intercept_
print('w1 = ', w1)
print('b1 = ', b1)


print('Best score for training data:', svm_model.best_score_,"\n") 

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n") 
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")



