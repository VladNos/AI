# Load libraries
import numpy as np
from pandas import read_csv
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from keras.layers import Layer, InputSpec
import keras.backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/5/wine.csv"
dataset = read_csv(url, header=0)
print(dataset.head(20))

df = pd.DataFrame(dataset)

dfmin = df.min()
dfmax = df.max()
normalized_df = (df - df.min())/(df.max() - df.min())
array = normalized_df.values

#for num_clusters in range(2,10): #daca nu stii cate clustere ar trebuii
#    clusterer = KMeans(n_clusters=num_clusters, n_jobs=4)
#    preds = clusterer.fit_predict(array)
#    score = silhouette_score (array, preds, metric='euclidean')
#    print ("For n_clusters = {}, Kmeans silhouette score is {})".format(num_clusters, score))

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(array)

def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1

    input_data = Input(shape=(dims[0],), name='input')
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks - 1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
    # latent hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
    x = encoded
    # internal layers of decoder
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
    # decoder output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    decoded = x

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model = Model(inputs=input_data, outputs=encoded, name='encoder')

    return autoencoder_model, encoder_model

n_epochs   = 100
batch_size = 128
dims = [array.shape[-1], 500, 500, 2000, 11] #ultima chestie e nr de coloane
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = n_epochs
batch_size = batch_size
save_dir = './resultsB' #sa modifici pt C
autoencoder, encoder = autoencoder(dims, init=init)

autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(array, array, batch_size=batch_size, epochs=pretrain_epochs)

autoencoder.save_weights(save_dir + '/ae_weights.h5')
autoencoder.load_weights(save_dir + '/ae_weights.h5')

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
loss = 0
index = 0
maxiter = 1000
update_interval = 100
tol = 0.001 # tolerance threshold to stop training
index_array = np.arange(array.shape[0])

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(array, verbose=0)
        p = target_distribution(q)
idx = index_array[index * batch_size: min((index+1) * batch_size, array.shape[0])]
loss = model.train_on_batch(x=array[idx], y=p[idx])
index = index + 1 if (index + 1) * batch_size <= array.shape[0] else 0

model.save_weights(save_dir + '/DEC_model_final.h5')
model.load_weights(save_dir + '/DEC_model_final.h5')

q = model.predict(array, verbose=0)
p = target_distribution(q)
y_pred = q.argmax(1) #daca crapa, schima in 2, ruleaza, pune 1 inapoi si o sa mearga, no ideaF

score = silhouette_score(array, y_pred, metric='euclidean')
print(score)

dimsTensor = Input(dims)
autoencoder.load_weights(save_dir + '/ae_weights.h5')
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(array))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
y_pred_last = np.copy(y_pred)
model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=pretrain_optimizer)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _  = model.predict(array, verbose=0)
        p = target_distribution(q)
        y_pred = q.argmax(1)
        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, array.shape[0])]
    loss = model.train_on_batch(x=array[idx], y=[p[idx], array[idx]])
    index = index + 1 if (index + 1) * batch_size <= array.shape[0] else 0

model.save_weights(save_dir + '/b_DEC_model_final.h5')
model.load_weights(save_dir + '/b_DEC_model_final.h5')

#B
info = [(8.1 - dfmin["fixed acidity"])/(dfmax["fixed acidity"] - dfmin["fixed acidity"]),
        (0.545 - dfmin["volatile acidity"])/(dfmax["volatile acidity"] - dfmin["volatile acidity"]),
        (0.18 - dfmin["citric acid"]) / (dfmax["citric acid"] - dfmin["citric acid"]),
        (1.9 - dfmin["residual sugar"]) / (dfmax["residual sugar"] - dfmin["residual sugar"]),
        (0.08 - dfmin["chlorides"]) / (dfmax["chlorides"] - dfmin["chlorides"]),
        (13 - dfmin["free sulfur dioxide"]) / (dfmax["free sulfur dioxide"] - dfmin["free sulfur dioxide"]),
        (35 - dfmin["total sulfur dioxide"]) / (dfmax["total sulfur dioxide"] - dfmin["total sulfur dioxide"]),
        (0.9972 - dfmin["density"]) / (dfmax["density"] - dfmin["density"]),
        (3.3 - dfmin["pH"]) / (dfmax["pH"] - dfmin["pH"]),
        (0.59 - dfmin["sulphates"]) / (dfmax["sulphates"] - dfmin["sulphates"]),
        (9.8 - dfmin["alcohol"]) / (dfmax["alcohol"] - dfmin["alcohol"])
        ]
aux = np.asarray(info)
aux = aux.reshape(1, 11) #daca da eroare, tre sa fie invers fata de cum zice ca trebe sa fie, nu intreba :))

q, _ = model.predict(aux, verbose=0)
p = target_distribution(q)
y_pred = q.argmax(1) #daca crapa, schimba in 2, apoi inapoi 1 si merge, still no idea
print(y_pred)

#pt C, scoti afara coloanele pe care nu ti le da cerinta

#https://towardsdatascience.com/deep-clustering-for-financial-market-segmentation-2a41573618cf