import pandas as pd
import numpy as np 
import random
from tensorflow.keras import layers, models
import scipy.io
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2DTranspose

data=scipy.io.loadmat('data_for_cnn.mat')
data=pd.DataFrame(data['ecg_in_window'])
labels=scipy.io.loadmat('class_label.mat')
labels=pd.DataFrame(labels['label'])

data = (data-data.mean(axis=0))/data.std(axis=0)
print(data.mean(),data.mean(axis=0))
w=np.random.rand(1000,3)

np.random.seed(1500)

train_rows=random.sample(range(0,labels.size), 70*labels.size//100)
train_rows.sort()
test_rows=[rows for rows in data.index.values if rows not in train_rows]

train_X = data.iloc[train_rows].values
train_y = labels.iloc[train_rows].values
test_X = data.iloc[test_rows].values
test_y = labels.iloc[test_rows].values

train_X=np.expand_dims(train_X,2)
test_X=np.expand_dims(test_X,2)

model=models.Sequential()
model.add(layers.Conv1D(filters=8, kernel_size=2,input_shape=(1000, 1),strides=4,activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dense(16))
model.add(layers.Lambda(lambda x: K.expand_dims(x, axis=2)))
model.add(layers.Conv2DTranspose(filters=8,kernel_size=1,strides=1))
model.add(layers.Reshape((1000,1)))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(train_X,train_X,validation_data=(test_X,test_X),epochs=400);

