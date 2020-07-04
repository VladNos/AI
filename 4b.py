# Load libraries
import numpy as np
from pandas import read_csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf



# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/4/tshirts.csv"
names = ['temperature', 'femaleTshirts', 'maleTshirts']
dataset = read_csv(url, names=names, header=0)

df = pd.DataFrame(dataset)
cols = [0, 2] #coloanele care ne intereseaza
df = df[df.columns[cols]]

dfmin = df.min()
print(dfmin)
dfmax = df.max()
normalized_df = (df - df.min())/(df.max() - df.min())
print(normalized_df)
array = normalized_df.values
X = array[:, (0)] #coloanele de input
y = array[:, 1].reshape(-1, 1) #coloanele de output
print(X.shape)
X_train = X[:100]
X_validation = X[50:]
Y_train = y[:100]
Y_validation = y[50:]

print(X_train)
print(Y_train)

# define the keras model
model = Sequential()
model.add(Dense(9, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the keras model
#la loss ai de ales intre mean_absolute_error, mean_squared_logarithmic_error si mean_squared_error
#la optimiser ii adam, SGD, adadelta, NAG, momentum
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, Y_train, epochs=32, batch_size=16)

# evaluate the keras model
loss, accuracy = model.evaluate(X_validation, Y_validation)
print(loss)


#B
info = [26, 26, 25, 25, 27, 27, 27, 27, 24, 25, 23, 27, 27, 22, 24, 27, 24, 23, 25, 27, 26, 22, 24, 24, 24, 25, 24, 25, 23, 25]
info = (info - dfmin.temperature)/(dfmax.temperature - dfmin.temperature)
aux = np.asarray(info)
aux = aux.reshape(-1, 1) #daca da eroare, tre sa fie invers fata de cum zice ca trebe sa fie, nu intreba :))

print(aux)
predictions = model.predict(aux)
print(sum(predictions * (dfmax.maleTshirts - dfmin.maleTshirts) + dfmin.maleTshirts))