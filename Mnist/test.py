import tensorflow as tf
from numpy import load
import matplotlib.pyplot as plt
import random

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

model=tf.keras.models.load_model('mnistlocal.h5')

rand=random.randint(0,len(y_test))
test_data=x_test[rand]
test_label=y_test[rand]
probs=model.predict(tf.expand_dims(test_data,axis=0))
preds=tf.argmax(probs,axis=1)
plt.imshow(test_data)
plt.title(f"Predicted class {int(preds)} Actual class {test_label}")
plt.show()