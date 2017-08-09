
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


X_train = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/X_train.npy')
X_valid = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/X_valid.npy')

Y_train = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/Y_train.npy')
Y_valid = np.load(file='/home/eramamur/jemmie_3exphg_danq_input_data/Y_valid.npy')


# In[ ]:


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
      
model.add(Bidirectional(LSTM(320, return_sequences=True)))
          
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(units=3))
model.add(Activation('sigmoid'))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.summary()


# In[ ]:


model.fit(x=X_train, y=Y_train, batch_size=100, epochs=60, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0)


# In[ ]:


model.save('/home/eramamur/danq_60_epochs.h5')


# In[ ]:


pred_train = model.predict_proba(X_train)
pred_valid = model.predict_proba(X_valid)


# In[ ]:


from sklearn.metrics import roc_auc_score
print "Training AUC:", roc_auc_score(Y_train, pred_train)
print "Validation AUC:", roc_auc_score(Y_valid, pred_valid)


# In[ ]:




