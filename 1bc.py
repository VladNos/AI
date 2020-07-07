# Load libraries
import numpy as np
from pandas import read_csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import math

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/1/homeData.csv"
dataset = read_csv(url, header=0)
df = pd.DataFrame(dataset)

df = df.iloc[0:150, :]  #primele 150 de randuri
cols = [2, 3, 4, 5, 6, 7] #coloanele care ne intereseaza
df = df[df.columns[cols]]

dfmin = df.min()
dfmax = df.max()
normalized_df = (df - df.min())/(df.max() - df.min())
array = normalized_df.values

X = array[:, (1, 2, 3, 4, 5)] #coloanele de input
y = array[:, 0].reshape(-1, 1) #coloanele de output

X_train = X[:100,:]
X_validation = X[50:,:]
Y_train = y[:100,:]
Y_validation = y[50:,:]

# define the keras model
model = Sequential()
model.add(Dense(9, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the keras model
#la loss ai de ales intre mean_absolute_error, mean_squared_logarithmic_error si mean_squared_error
#la optimiser ii adam, SGD, adadelta, NAG, momentum
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, Y_train, epochs=100, batch_size=8)

# evaluate the keras model
loss, accuracy = model.evaluate(X_validation, Y_validation)
print(math.sqrt(loss)* (dfmax.price - dfmin.price) + dfmin.price)
print('Accuracy: %.2f' % (accuracy*100))

#C
info = [(3 - dfmin.bedrooms)/(dfmax.bedrooms - dfmin.bedrooms),
        (1 - dfmin.bathrooms)/(dfmax.bathrooms - dfmin.bathrooms),
        (1780 - dfmin.sqft_living)/(dfmax.sqft_living - dfmin.sqft_living),
        (7460 - dfmin.sqft_lot)/(dfmax.sqft_lot - dfmin.sqft_lot),
        (1 - dfmin.floors)/(dfmax.floors - dfmin.floors)]
aux = np.asarray(info)
aux = aux.reshape(1, 5) #daca da eroare, tre sa fie invers fata de cum zice ca trebe sa fie, nu intreba :))

predictions = model.predict(aux)
print(predictions * (dfmax.price - dfmin.price) + dfmin.price)