# Load libraries
import numpy as np
from pandas import read_csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/3/persons.csv"
dataset = read_csv(url, header=0)

df = pd.DataFrame(dataset)
df = df.iloc[0:150, :]  #primele 150 de randuri

le = preprocessing.LabelEncoder()
le.fit(df["sex"])
df["sex"] = le.transform(df["sex"])

dfmin = df.min()
dfmax = df.max()
normalized_df = (df - df.min())/(df.max() - df.min())
array = normalized_df.values

X = array[:, (0, 1, 2)] #cate randuri vrei sa iei (pt train + validare) : coloanele de input
y = array[:, 3].reshape(-1, 1) #coloanele de output
X_train = X[:100, :]
X_validation = X[50:, :]
Y_train = y[:100, :]
Y_validation = y[50:, :]

# define the keras model
model = Sequential()
model.add(Dense(9, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the keras model
#la loss ai de ales intre mean_absolute_error, mean_squared_logarithmic_error si mean_squared_error
#la optimiser ii adam, SGD, adadelta, NAG, momentum
model.compile(loss='mse', optimizer='SGD', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, Y_train, epochs=100, batch_size=8)

# evaluate the keras model
loss, accuracy = model.evaluate(X_validation, Y_validation)
print("average Loss: " + loss)
print('Accuracy: %.2f' % (accuracy*100))

#C  find and replace ctrl r
info = [(47 - dfmin.age)/(dfmax.age - dfmin.age), #vezi sa fie in ordinea din csv, nu cum ti le zice profa in pdf
        (170 - dfmin.height)/(dfmax.height - dfmin.height),
        (100 - dfmin.weight)/(dfmax.weight - dfmin.weight)]
aux = np.asarray(info)
aux = aux.reshape(-1, 3) #daca da eroare, tre sa fie invers fata de cum zice ca trebe sa fie, nu intreba :))

predictions = model.predict(aux)
if predictions < 0.5:
    print("female")
else:
    print("male")

#pt C schimbi numa csv-u