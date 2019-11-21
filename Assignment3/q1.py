import pandas as pd
import numpy as np 
import random
from tensorflow.keras import layers, models
import scipy.io
import matplotlib.pyplot as plt

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
model.add(layers.Conv1D(filters=50, kernel_size=15 ,strides=2,     
              input_shape=(1000, 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=20, kernel_size=10 ,strides=2))
model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(40, activation= 'relu' ))
model.add(layers.Dense(20, activation= 'relu' ))
model.add(layers.Dense(1, activation= 'relu' ))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(train_X,train_y, epochs=200,validation_data=(test_X,test_y),use_multiprocessing=True)

y_pred=model.predict(test_X.astype(float))
y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0

acc = 0

for i in range(test_y.shape[0]):
	if(y_pred[i] == test_y[i]):
		acc += 1

acc /= test_y.shape[0]

print(acc)