from numpy import load
import tensorflow as tf
import matplotlib.pyplot as plt


mnist=load('mnist.npz')
data=mnist.files
for i,j in enumerate(data):
    if i==0:
        x_test=mnist[j]
    elif i==1:
        x_train=mnist[j]
    elif i==2:
        y_train=mnist[j]
    else:
        y_test=mnist[j]

model_1=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")

])
model_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_1.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
model_1.save('mnistlocal.h5')