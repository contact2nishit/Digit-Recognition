#!/usr/bin/env python
# coding: utf-8

# ### Dataset: Kaggle Dataset

# ### Technique: Keras for Digit Recognition

# In[2]:


'''Project Docstring'''


# In[3]:


#pip install tensorflow


# In[22]:


#Importing Libraries
from tensorflow.keras.datasets import mnist
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold


# In[2]:


#loading dataset from the library
(trainx, trainy), (testx, testy) = mnist.load_data()


# In[3]:


#summarize loaded dataset
print(f"Train: x = {trainx.shape}, y = {trainy.shape}")
print(f"Test: x = {testx.shape}, y = {testy.shape}")


# In[4]:


#plotting first few images
for i in range(9):
    plt.subplot(3,3,1+i)
    plt.imshow(trainx[i],cmap = plt.get_cmap('gray'))
plt.show()


# In[5]:


trainx = trainx.reshape((-1,28,28,1))
testx = testx.reshape((-1,28,28,1))


# In[6]:


print(trainy)


# In[7]:


trainy = to_categorical(trainy)
print(trainy.shape)
testy = to_categorical(testy)
print(testy.shape)


# In[8]:


trainx = trainx.astype('float32')
testx = testx.astype('float32')
trainx = trainx/255.0
testx = testx/255.0


# In[19]:


#Defining model
model = Sequential()
model.add(Conv2D(32,(3,3), activation = relu, kernel_initializer = 'he_uniform',input_shape = (28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation = relu, kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = softmax))
#Compiliing the model
optim = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])


# In[28]:


#Evaluating model
scores, histories = [],[]
kfold = KFold(5, shuffle = True, random_state = 1)
for train_ix, test_ix in kfold.split(trainx):
    # select rows for train and test
    trainX, trainY, testX, testY = trainx[train_ix], trainy[train_ix], trainx[test_ix], trainy[test_ix]
    # fit model
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
    # evaluate model
    acc = model.evaluate(testX, testY, verbose=0)
    print(acc)
    scores.append(acc[1]*100)
    histories.append(history)


# In[48]:


acc_test = model.evaluate(testx,testy, verbose = 0)
print(acc_test)
print(model.metrics_names)


# In[53]:


Y_pred = model.predict(testx)


# In[37]:


print(testX.shape)
print(testx.shape)
print(testY.shape)
print(testy.shape)


# In[56]:


print(Y_pred.shape)
print(Y_pred[3])
print(testy[3])
plt.imshow(testx[3],cmap = plt.get_cmap('gray'))
plt.show()


# In[30]:


for i in range(len(histories)):
 # plot loss
 plt.subplot(2, 1, 1)
 plt.title('Cross Entropy Loss')
 plt.plot(histories[i].history['loss'], color='blue', label='train')
 plt.plot(histories[i].history['val_loss'], color='orange', label='test')
 plt.legend()
 # plot accuracy
 plt.subplot(2, 1, 2)
 plt.title('Classification Accuracy')
 plt.plot(histories[i].history['accuracy'], color='blue', label='train')
 plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
 plt.show()


# In[ ]:




