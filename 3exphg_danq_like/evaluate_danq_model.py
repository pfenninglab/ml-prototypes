
# coding: utf-8

# In[11]:


from keras.models import Sequential


from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM

from keras.utils import plot_model




model = Sequential()

model.add(Conv1D(input_shape = (1000, 4),
                    padding="valid",
                    strides=1,
                    activation="relu",
                    kernel_size=26,
                    filters=320))                 
                 
          
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
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.summary()


# In[12]:


model.load_weights('/home/eramamur/danq_bestmodel_3exphg_checkpoint.hdf5')


# In[19]:


import numpy as np
X_train = np.load('/home/eramamur/jemmie_3exphg_danq_input_data/X_train.npy')
Y_train = np.load('/home/eramamur/jemmie_3exphg_danq_input_data/Y_train.npy')

X_valid = np.load('/home/eramamur/jemmie_3exphg_danq_input_data/X_valid.npy')
Y_valid = np.load('/home/eramamur/jemmie_3exphg_danq_input_data/Y_valid.npy')


# In[20]:


pred_train = model.predict_proba(X_train)
pred_valid = model.predict_proba(X_valid)


# In[21]:


from sklearn.metrics import roc_auc_score

label_names = ['G', 'M', 'N']


for i in xrange(len(label_names)):
    print "Training AUC:", roc_auc_score(Y_train[:,i], pred_train[:,i])
    print "Validation AUC:", roc_auc_score(Y_valid[:,i], pred_valid[:,i])

