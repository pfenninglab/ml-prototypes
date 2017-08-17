
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


X_train = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/X_train.npy')
X_valid = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/X_valid.npy')

Y_train = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/Y_train.npy')
Y_valid = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/Y_valid.npy')

#X_train = X_train[0:20][:][:]
#Y_train = Y_train[0:20][:]

#X_valid = X_train[0:20][:][:]
#Y_valid = Y_valid[0:20][:][:]

# In[ ]:

import h5py

danq_original = h5py.File('/home/eramamur/Github_repos/ml-prototypes/3exphg_danq_like/DanQ_bestmodel.hdf5')

conv_layer_kernel_weights = danq_original['layer_0']['param_0']
conv_layer_kernel_weights = np.transpose(conv_layer_kernel_weights, (2,1,0,3))[:,:,:,0]
conv_layer_bias_weights = np.array(danq_original['layer_0']['param_1'])

conv_layer_weights = [conv_layer_kernel_weights, conv_layer_bias_weights]


from keras.models import Sequential


from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers


from keras.utils import plot_model


model = Sequential()

conv_layer = Conv1D(input_shape = (1000, 4),
                    padding="valid",
                    strides=1,
                    activation="relu",
                    kernel_size=26,
                    filters=320)



model.add(conv_layer)                 
                 
          
model.add(MaxPooling1D(pool_size=13, strides=13))

model.add(Dropout(0.2))

brnn = Bidirectional(LSTM(320, return_sequences=True))

model.add(brnn)
          
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(units=3))
model.add(Activation('sigmoid'))

print 'compiling model'
rmsprop = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy', optimizer=rmsprop)

model.load_weights('/home/eramamur/Github_repos/ml-prototypes/3exphg_danq_like/danq_bestmodel_3exphg.hdf5')
conv_layer.set_weights(conv_layer_weights)

model.summary()


# In[ ]:


checkpointer = ModelCheckpoint(filepath="/home/eramamur/danq_bestmodel_3exphg_checkpoint.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)



model.fit(x=X_train, y=Y_train, batch_size=100, epochs=60, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer,earlystopper])


# In[ ]:


pred_train = model.predict_proba(X_train)
pred_valid = model.predict_proba(X_valid)


# In[ ]:


from sklearn.metrics import roc_auc_score

label_names = ['G', 'M', 'N']


for i in xrange(len(label_names)):
    print "Training AUC:", roc_auc_score(Y_train[:,i], pred_train[:,i])
    print "Validation AUC:", roc_auc_score(Y_valid[:,i], pred_valid[:,i])

# In[ ]:




