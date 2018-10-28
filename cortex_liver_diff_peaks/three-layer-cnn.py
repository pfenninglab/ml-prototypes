import argparse
import numpy as np
from keras.models import Sequential


from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

from keras import optimizers

def get_three_layer_cnn_model(numLabels, numConvFilters_1=320, numConvFilters_2=480, numConvFilters_3=960, pool1_dropout=0.2, pool2_dropout=0.2, conv3_dropout=0.5, learningRate=0.001):
    model = Sequential()

    conv_layer_1 = Conv1D(input_shape = (1000, 4),
                        padding="valid",
                        strides=1,
                        activation="relu",
                        kernel_size=8,
                        filters=numConvFilters_1)



    model.add(conv_layer_1)                 

    model.add(MaxPooling1D(pool_size=4, strides=4))

    model.add(Dropout(pool1_dropout))

	conv_layer_2 = Conv1D(input_shape = (1000, 4),
					padding="valid",
					strides=1,
					activation="relu",
					kernel_size=8,
					filters=numConvFilters_2)
	
	model.add(conv_layer_2)
	
	model.add(MaxPooling1D(pool_size=4, strides=4))
	model.add(Dropout(pool2_dropout))

	conv_layer_3 = Conv1D(input_shape = (1000, 4),
					padding="valid",
					strides=1,
					activation="relu",
					kernel_size=8,
					filters=numConvFilters_3)
	
	model.add(conv_layer_3)
	
	model.add(Dropout(conv3_dropout))
	
    model.add(Flatten())

    model.add(Dense(units=925))
    model.add(Activation('relu'))

    model.add(Dense(units=numLabels))
    model.add(Activation('sigmoid'))
    
    optim = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model

def train_cnn_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize=100,
                     numEpochs=60,
					 numConvFilters_1=320,
					 numConvFilters_2=480,
					 numConvFilters_3=960,
					 pool1_dropout=0.2,
					 pool2_dropout=0.2,
					 conv3_dropout=0.5,
					 learningRate=0.001
                    ):
    numLabels = 1
    model = get_three_layer_cnn_model(numLabels, numConvFilters_1, numConvFilters_2, numConvFilters_3, pool1_dropout, pool2_dropout, conv3_dropout, learningRate)
    model.summary()
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True)

    model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid),
              initial_epoch=0, callbacks=[checkpointer])
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a three layer CNN model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data', required=True)
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels', required=True)
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data', required=True)
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels', required=True)
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False)
    parser.add_argument('-c1', '--num-conv-filters-1', type=int, help='number of first layer convolutional filters', required=False)
    parser.add_argument('-c2', '--num-conv-filters-2', type=int, help='number of second layer convolutional filters', required=False)
    parser.add_argument('-c1', '--num-conv-filters-3', type=int, help='number of third layer convolutional filters', required=False)
    parser.add_argument('-p1drop', '--pool-1-dropout-rate', type=float, help='dropout rate for pooling layer 1', required=False)
    parser.add_argument('-p2drop', '--pool-2-dropout-rate', type=float, help='dropout rate for pooling layer 2', required=False)
    parser.add_argument('-c3drop', '--conv-3-dropout-rate', type=float, help='dropout rate for convolutional layer 3', required=False)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for optimizer', required=False)
    
    args = parser.parse_args()
    
    X_train = np.load(file=args.xtrain)
    Y_train = np.load(file=args.ytrain)
    X_valid = np.load(file=args.xvalid)
    Y_valid = np.load(file=args.yvalid)
    
    train_cnn_model(modelOut=args.model_out,
                     X_train=X_train,
                     Y_train=Y_train,
                     X_valid=X_valid,
                     Y_valid=Y_valid,
					 args.batch_size,
					 args.num_epochs,
					 args.num_conv_filters_1,
					 args.num_conv_filters_2,
					 args.num_conv_filters_3,
					 args.pool_1_dropout_rate,
					 args.pool_2_dropout_rate,
					 args.conv_3_dropout_rate,
					 args.learning_rate
                    )    
