# Load libraries
import numpy as np
from pandas import read_csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn import preprocessing



# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/4/tshirtsNew.csv"
dataset = read_csv(url, header=0)

df = pd.DataFrame(dataset)
cols = [0, 2, 3, 4] #coloanele care ne intereseaza
df = df[df.columns[cols]]

le = preprocessing.LabelEncoder()
le.fit(df["location"])
df["location"] = le.transform(df["location"])

le2 = preprocessing.LabelEncoder()
le2.fit(df["competitions"])
df["competitions"] = le2.transform(df["competitions"])



dfmin = df.min()
print(dfmin)
dfmax = df.max()
print(dfmax)
normalized_df = (df - df.min())/(df.max() - df.min())
print(normalized_df)
array = normalized_df.values
X = array[:, (0, 2, 3)] #coloanele de input
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
model.add(Dense(9, input_dim=3, activation='relu'))
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


#C

info = [(25 - dfmin.temperature)/(dfmax.temperature - dfmin.temperature), le2.transform(["many"]), le.transform(["high-school"])]
aux = np.asarray(info)
aux = aux.reshape(-1, 3) #daca da eroare, tre sa fie invers fata de cum zice ca trebe sa fie, nu intreba :))

print(aux)
predictions = model.predict(aux)
print(sum(predictions * (dfmax.maleTshirts - dfmin.maleTshirts) + dfmin.maleTshirts))