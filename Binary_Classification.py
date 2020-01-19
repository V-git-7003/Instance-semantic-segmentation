#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from Data_Preparation import *

X,y = create_training_data()


#def binary_classification_model():
model = Sequential()
model.add(Conv2D(64,(3,3),padding = 'same',input_shape=(X.shape[1:]),kernel_initializer = 'he_normal'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3),padding = 'same',kernel_initializer = 'he_normal'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu")) # added a new activation layer

model.add(Dense(1))
model.add(Activation('sigmoid')) # chaging from sigmoid to softmax to include more classes

#filepath="weights.best.hdf5"
ada = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ["accuracy"])

lr_model_history=model.fit(X,y,batch_size=10,epochs= 10,validation_split=0.1)


# ### 1. The learning rate for Adam was tested from 0.001 to 0.010 giving no model that gives good test accuracy.
# ### 2. Although the training and validation accuracy reaches 0.99 and 1. It still does not perform well on test data (3/6)

Test_D, img_names = Test_Data()
scores = model.predict_classes(Test_D)
print(Test_D.shape)
for i in range(len(Test_D)):
     print("Image:",img_names[i],"with score:",scores[i])

