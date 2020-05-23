#import all libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#get the mnist dataset from keras
mnist = tf.keras.datasets.mnist


#divide the dataset into training and testing
(x_train,y_train),(x_test,y_test)=mnist.load_data()


#normalize
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)


#build the neural network
#input layers
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#define parameters for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#train the model
model.fit(x_train, y_train,epochs=3)


#calculate validation loss and validation accuracy(out of sample)
val_loss, val_acc=model.evaluate(x_test, y_test)


#Testing
predictions=model.predict(np.array(x_test))
print(np.argmax(predictions[1]))


#Checking
plt.imshow(x_test[1])
plt.show()

