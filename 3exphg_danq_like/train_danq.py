
# coding: utf-8

# In[2]:


import numpy as np
import h5py
import scipy.io
np.random.seed(1337)


# In[3]:


trainmat = h5py.File('/projects/pfenninggroup/machineLearningForComputationalBiology/deepsea_train/train.mat')
validmat = scipy.io.loadmat('/projects/pfenninggroup/machineLearningForComputationalBiology/deepsea_train/valid.mat')
testmat = scipy.io.loadmat('/projects/pfenninggroup/machineLearningForComputationalBiology/deepsea_train/test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
Y_train = np.array(trainmat['traindata']).T


X_valid = np.transpose(validmat['validxdata'],axes=(0,2,1))
Y_valid = validmat['validdata']

#X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
#Y_test = testmat['testdata']


# In[4]:


#X_train = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/X_train.npy')
#X_valid = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/X_valid.npy')

#Y_train = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/Y_train.npy')
#Y_valid = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/Y_valid.npy')


# In[5]:


#X_train = X_train[0:20][:][:]
#Y_train = Y_train[0:20][:]

#X_valid = X_valid[0:20][:][:]
#Y_valid = Y_valid[0:20][:][:]

#X_test = X_test[0:20][:][:]
#Y_test = Y_test[0:20][:][:]


# In[6]:


#import h5py

#danq_original = h5py.File('/home/eramamur/Github_repos/ml-prototypes/3exphg_danq_like/DanQ_bestmodel.hdf5')

#conv_layer_kernel_weights = danq_original['layer_0']['param_0']
#conv_layer_kernel_weights = np.transpose(conv_layer_kernel_weights, (2,1,0,3))[:,:,:,0]
#conv_layer_bias_weights = np.array(danq_original['layer_0']['param_1'])

#conv_layer_weights = [conv_layer_kernel_weights, conv_layer_bias_weights]


# In[7]:


from keras.models import Sequential


from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import optimizers



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

model.add(Dense(units=919))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[8]:


checkpointer = ModelCheckpoint(filepath="/home/eramamur/Github_repos/ml-prototypes/3exphg_danq_like/current_job_checkpoint.hdf5", verbose=1, save_best_only=True)

model.fit(x=X_train, y=Y_train, batch_size=100, epochs=60, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer])


# In[ ]:




