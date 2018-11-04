import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
# download and read mnist
mnist = fetch_mldata('MNIST original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target


# split data to train and test (for faster calculation, just use 1/10 data)# split 
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

cls1 = LogisticRegression()
cls1.fit(X_train,Y_train)
y_train1=cls1.predict(X_train)
y_test1=cls1.predict(X_test)

train_accuracy=metrics.accuracy_score(Y_train,y_train1)
test_accuracy=metrics.accuracy_score(Y_test,y_test1)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

cls2 = BernoulliNB()
cls2.fit(X_train,Y_train)
y_train2=cls2.predict(X_train)
y_test2=cls2.predict(X_test)

train_accuracy=metrics.accuracy_score(Y_train,y_train2)
test_accuracy=metrics.accuracy_score(Y_test,y_test2)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

cls3 = LinearSVC()
cls3.fit(X_train,Y_train)
y_train3=cls3.predict(X_train)
y_test3=cls3.predict(X_test)

train_accuracy=metrics.accuracy_score(Y_train,y_train3)
test_accuracy=metrics.accuracy_score(Y_test,y_test3)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

param_grid = {"C":[0.009,0.01,0.011,0.1,0.5,0.8,1,1.2,1.3,1.5,1.7,5,10,100]}
print("Parameters:{}".format(param_grid))

grid_search = GridSearchCV(LinearSVC(),param_grid,cv=5)

grid_search.fit(X_train,Y_train)
print("Train set score:{:.4f}" .format(grid_search.score(X_train,Y_train)))
print("Test set score:{:.4f}" .format(grid_search.score(X_test,Y_test)))
print("Best parameters:{}" .format(grid_search.best_params_))
print("Best score on train set:{:.2f}"  .format(grid_search.best_score_))
