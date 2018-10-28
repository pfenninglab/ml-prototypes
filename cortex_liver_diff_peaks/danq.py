import argparse
import numpy as np
from keras.models import Sequential


from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import optimizers

def get_danq_model(numLabels, numConvFilters=320, poolingDropout=0.2, brnnDropout=0.5, learningRate=0.001):
    model = Sequential()

    conv_layer = Conv1D(input_shape = (1000, 4),
                        padding="valid",
                        strides=1,
                        activation="relu",
                        kernel_size=26,
                        filters=numConvFilters)



    model.add(conv_layer)                 


    model.add(MaxPooling1D(pool_size=13, strides=13))

    model.add(Dropout(poolingDropout))

    brnn = Bidirectional(LSTM(320, return_sequences=True))

    model.add(brnn)

    model.add(Dropout(brnnDropout))

    model.add(Flatten())

    model.add(Dense(input_dim=75*640, units=925))
    model.add(Activation('relu'))

    model.add(Dense(units=numLabels))
    model.add(Activation('sigmoid'))
    
    optim = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model

def train_danq_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize=100,
                     numEpochs=60,
                     numConvFilters=320,
                     poolingDropout=0.2,
                     brnnDropout=0.5,
                     learningRate=0.001
                    ):
    numLabels = 1
    model = get_danq_model(numLabels, numConvFilters, poolingDropout, brnnDropout, learningRate)
    model.summary()
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=60, verbose=0, mode='auto')
    model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid),
              initial_epoch=0, callbacks=[checkpointer, earlystopper])
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a DanQ model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data', required=True)
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels', required=True)
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data', required=True)
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels', required=True)
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=60)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use', required=False, default=320)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-bdrop', '--brnn-dropout-rate', type=float, help='dropout rate for B-LSTM layer', required=False, default=0.5)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for rmsprop optimizer', required=False, default=0.001)
    
    args = parser.parse_args()
    
    X_train = np.load(file=args.xtrain)
    Y_train = np.load(file=args.ytrain)
    X_valid = np.load(file=args.xvalid)
    Y_valid = np.load(file=args.yvalid)
    
    train_danq_model(modelOut=args.model_out,
                     X_train=X_train,
                     Y_train=Y_train,
                     X_valid=X_valid,
                     Y_valid=Y_valid,
                     batchSize=args.batch_size,
                     numEpochs=args.num_epochs,
                     numConvFilters=args.num_conv_filters,
                     poolingDropout=args.pool_dropout_rate,
                     brnnDropout=args.brnn_dropout_rate,
                     learningRate=args.learning_rate
                    )    
