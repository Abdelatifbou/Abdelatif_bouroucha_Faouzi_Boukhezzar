import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Normalization
from tensorflow.python.ops.initializers_ns import variables
from tensorflow.keras.losses import CategoricalCrossentropy

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
mm = tf.keras.models.Sequential()
mm.add(Normalization())
mm.add(Flatten(input_shape=(28, 28)))
mm.add(Dense(100, activation='relu'))
mm.add(Dense(20, activation='relu'))
mm.add(Dense(10, activation='softmax'))
mm.build(input_shape=(None, 28, 28))
mm.summary()

batch_size = 32
x_batch = x_train[:batch_size]
y_batch = y_train[:batch_size]

with tf.GradientTape() as tape:
    y_pred = mm(x_batch)
    loss = CategoricalCrossentropy()(to_categorical(y_batch, num_classes=10), y_pred)
grad = tape.gradient(loss, mm.trainable_variables)
for g in grad:
    print(g.numpy())
plt.imshow(x_train[10])
plt.show()
print(x_train[0])