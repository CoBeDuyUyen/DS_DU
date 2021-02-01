# -*- coding: utf-8 -*-
"""FruitClassificationResNet50.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xb7T_tCCbaX3zoAGUHHkujnj-ZQhA0IQ

# Read the data
"""

from google.colab import files 
files.upload()
!pip install -q kaggle
!mkdir -p ~/.kaggle 
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d moltean/fruits
!ls

!unzip fruits.zip #unzip the file

"""# Exploring the data"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd 
import numpy as np 
import os 
from keras import losses 
from keras import optimizers 
from keras import metrics 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.resnet50 import preprocess_input

train_path = "fruits-360/Training"
test_path = "fruits-360/Test"

"""# ResNet50"""

from keras.applications.resnet50 import ResNet50
rn50 = ResNet50()
print(rn50.summary())

from tensorflow.keras import models, layers
input_layer=layers.Input(shape=(100,100,3))
resnet_model=ResNet50(input_tensor=input_layer,include_top=False)
resnet_model.summary()

last_layer=resnet_model.output
flatten=layers.Flatten()(last_layer) 
output_layer=layers.Dense(131,activation='softmax')(flatten)
model=models.Model(inputs=input_layer,outputs=output_layer)
model.summary()

for layer in model.layers[:-1]:
    layer.trainable=False
model.summary()

model.compile(loss = 'categorical_crossentropy',  
   optimizer = keras.optimizers.SGD(learning_rate = 0.01), metrics = [metrics.categorical_accuracy])

train_data = ImageDataGenerator().flow_from_directory(train_path, target_size = (100,100), shuffle=True)
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size = (100,100), shuffle=True)

hist = model.fit_generator(train_data,
          epochs = 10, 
          shuffle=True,
          validation_data = test_data)

import matplotlib.pyplot as plt
# loss
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(hist.history['accuracy'], label="train acc")
plt.plot(hist.history['val_accuracy'], label="val acc")
plt.legend()
plt.show()

"""# Prediction"""

label = list(test_data.class_indices.items())
print(label)

import matplotlib.image as mpimg
img=mpimg.imread('mango.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

from keras.preprocessing.image import load_img
img = load_img('mango.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict_on_batch(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[88])

print('probability of the answer: ',model.predict(img)[0, 88])
print('probability of the correct answer: ',model.predict(img)[0, 64])

img=mpimg.imread('tomato.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('tomato.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[17])

print('probability of the answer: ',model.predict(img)[0, 17])
print('probability of the correct answer: ',model.predict(img)[0, 120], ', ', model.predict(img)[0, 121], ', ', model.predict(img)[0, 122], ', ', model.predict(img)[0, 123])

img=mpimg.imread('banana.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('banana.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict_on_batch(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[17])

print('probability of the correct answer: ',model.predict(img)[0, 16])

img=mpimg.imread('kiwi.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('kiwi.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict_on_batch(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[36])

print('probability of the answer: ',model.predict(img)[0, 36])
print('probability of the correct answer: ',model.predict(img)[0, 56])

img=mpimg.imread('orange.png', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('orange.png')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict_on_batch(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[97])

print('probability of the answer: ',model.predict(img)[0, 97])
print('probability of the correct answer: ',model.predict(img)[0, 77])

img=mpimg.imread('cucumber.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('cucumber.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict_on_batch(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[35])

print('probability of the answer: ',model.predict(img)[0, 35])
print('probability of the correct answer: ',model.predict(img)[0, 37])

img=mpimg.imread('strawberry.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('strawberry.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[116])

print('probability of the correct answer: ',model.predict(img)[0, 116])

img=mpimg.imread('watermelon.jpg', 0)
imgplot = plt.imshow(img)
plt.show()

img = load_img('watermelon.jpg')
img = img.resize((100, 100))
img = img_to_array(img) 

img = img.reshape( -1,100, 100,3)

predict = model.predict(img)
predict = predict.argmax(axis=-1)
print(predict)

print(label[68])

print('probability of the answer: ',model.predict(img)[0, 68])
print('probability of the correct answer: ',model.predict(img)[0, 130])