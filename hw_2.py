import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
import numpy as np

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10

# download and load the data (split them between train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# expand the channel dimension
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0],784)
input_shape = (784,)

# make the value of pixels from [0, 255] to [0, 1] for further process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
  
# define the model
model = Sequential()
model.add(Dense(500,input_shape=input_shape)) # 输入层
model.add(Activation('tanh')) 
model.add(Dropout(0.5)) 

model.add(Dense(500)) 
model.add(Activation('tanh'))
model.add(Dropout(0.5))
 
model.add(Dense(500))
model.add(Activation('tanh'))

model.add(Dense(10)) # 输出层
model.add(Activation('softmax')) 

# define the object function, optimizer and metrics
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# train
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, shuffle=True, verbose=2, validation_split=0.3)

scores = model.evaluate(x_test,y_test, batch_size=BATCH_SIZE, verbose=0)# 误差评价
print("The loss is %f" % scores)
 
 
result = model.predict(x_train,batch_size=BATCH_SIZE,verbose=1)
result_max = np.argmax(result, axis = 1)
test_max = np.argmax(y_train, axis = 1)
result_bool = np.equal(result_max, test_max) 
true_num = np.sum(result_bool) 
print("The train accuracy of the model is %f" % (true_num/len(result_bool)))

result1 = model.predict(x_test,batch_size=BATCH_SIZE,verbose=1)
result_max1 = np.argmax(result1, axis = 1) 
test_max1 = np.argmax(y_test, axis = 1)
result_bool1 = np.equal(result_max1, test_max1) 
true_num1 = np.sum(result_bool1) 
print("The text accuracy of the model is %f" % (true_num1/len(result_bool1)))


