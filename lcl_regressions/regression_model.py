import argparse
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from scipy.stats import pearsonr, spearmanr
from keras import optimizers
import keras.backend as K
import keras.metrics
from keras import regularizers

def pearson_correlation(x, y):
    n = K.sum(K.ones_like(x))
    sum_x = K.sum(x)
    sum_y = K.sum(y)
    
    sum_x_sq = K.sum(K.square(x))
    sum_y_sq = K.sum(K.square(y))
    
    psum = K.sum(x * y)
    num = psum - (sum_x * sum_y / n)
    den = K.sqrt((sum_x_sq - K.square(sum_x) / n) *  (sum_y_sq - K.square(sum_y) / n))
    
    r = num / den

    return r


keras.metrics.pearson_correlation = pearson_correlation


def get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length, kernelSize,regularization):
    model = Sequential()
	
    conv1_layer = Conv1D(input_shape=(length, 4),
                        padding="valid",
                        strides=1,
                        activation="relu",
                        kernel_size=kernelSize, #changed to kernelSize, originally set manually to 8
                        filters=1000,
                        use_bias=True)


    model.add(conv1_layer)
    reg = "None"
    if regularization=='l1':
        reg=regularizers.l1(10**-6)
    elif regularization=='l2':
        reg=regularizers.l2(10**-6)
    
    for i in range(numConvLayers-1):
        if reg=="None":
            convn_layer = Conv1D(padding="valid",
                            strides=1,
                            activation="relu",
                            kernel_size=kernelSize,
                            filters=numConvFilters,
                            use_bias=True)
        else:
            convn_layer = Conv1D(padding="valid",
                            strides=1,
                            activation="relu",
                            kernel_size=kernelSize,
                            filters=numConvFilters,
                            use_bias=True,
                            activity_regularizer=reg)
        model.add(convn_layer)
    

    model.add(MaxPooling1D(pool_size=13, strides=13))

    model.add(Dropout(poolingDropout))

    model.add(Flatten())

    model.add(Dense(units=preLastLayerUnits, use_bias=True))
    model.add(Activation('sigmoid'))

    model.add(Dense(units=numLabels, use_bias=True))
    
    return model

def train_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize,
                     numEpochs,
                     numConvLayers,
                     numConvFilters,
                     preLastLayerUnits,
                     poolingDropout,
                     learningRate,
                     momentum,
                     length,
                     pretrainedModel,kernelSize,regularization
                    ):
    numLabels = Y_train.shape[1]
    if pretrainedModel:
        model = load_model(pretrainedModel)
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length, kernelSize, regularization)

    optim = optimizers.SGD(lr=learningRate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=optim, metrics=[pearson_correlation])    
    model.summary()
    
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=60, verbose=0, mode='min')
    print(X_train.shape)
    print(Y_train.shape)
    model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer, earlystopper])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a convolutional neural network model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data', required=True)
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels', required=True)
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data', required=True)
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels', required=True)
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=60)
    parser.add_argument('-n', '--num-conv-layers', type=int, help='number of convolutional layers to use', required=False, default=4)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use in layers after the first one', required=False, default=100)
    parser.add_argument('-u', '--pre-last-layer-units', type=int, help='number of sigmoid units in the layer before the output layer', required=False, default=100)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.00)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=1000)
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model', required=False, default=None)
    parser.add_argument('-ks','--kernel-size',type=int,help="Kernel size",required=False,default=8)
    parser.add_argument('-r','--regularization',type=str,help="regularization",required=False,default="None")   
    args = parser.parse_args()
    
    X_train = np.load(file=args.xtrain)
    Y_train = np.load(file=args.ytrain)
    X_valid = np.load(file=args.xvalid)
    Y_valid = np.load(file=args.yvalid)

    train_model(modelOut=args.model_out,
                     X_train=X_train,
                     Y_train=Y_train,
                     X_valid=X_valid,
                     Y_valid=Y_valid,
                     batchSize=args.batch_size,
                     numEpochs=args.num_epochs,
                     numConvLayers=args.num_conv_layers,
                     numConvFilters=args.num_conv_filters,
                     preLastLayerUnits=args.pre_last_layer_units,
                     poolingDropout=args.pool_dropout_rate,
                     learningRate=args.learning_rate,
                     momentum=args.momentum,
                     length=args.length,
                     pretrainedModel=args.pretrained_model,
                     kernelSize=args.kernel_size,
                     regularization=args.regularization
                    )
