import argparse
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import auc,roc_curve,roc_auc_score, precision_recall_curve, average_precision_score, auc, balanced_accuracy_score, accuracy_score

from keras import optimizers
import keras.backend as K

def specificity_metric(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def auRoc(Y, pred):
    fpr, tpr, thresholds = roc_curve(Y, pred, pos_label=1)
    auRoc = auc(fpr, tpr)
    return auRoc, fpr, tpr, thresholds

def auPrc(Y, pred):
    precision, recall, thresholds = precision_recall_curve(Y,pred)
    auPrc = auc(recall, precision)
    return auPrc, precision, recall, thresholds

def balancedAccuracy(Y, pred):
    pred_binary = pred>0.5
    balancedAccuracy = balanced_accuracy_score(Y, pred_binary)
    return balancedAccuracy

def sensitivity(Y, pred):
    pred_binary = pred>0.5

    positiveClassTrue = []
    positiveClassPred = []
    
    for i, val in enumerate(Y):
        currTrueLabel = Y[i]
        currPred = pred_binary[i]
        if currTrueLabel == 1:
            positiveClassTrue.append(currTrueLabel)
            positiveClassPred.append(currPred)          

            
    positiveClassTrue = np.array(positiveClassTrue)
    positiveClassPred = np.array(positiveClassPred)    
    positiveAccuracy = accuracy_score(positiveClassTrue, positiveClassPred)
    
    return positiveAccuracy


def specificity(Y, pred):
    pred_binary = pred>0.5


    negativeClassTrue = []
    negativeClassPred = []    

    for i, val in enumerate(Y):
        currTrueLabel = Y[i]
        currPred = pred_binary[i]
        if currTrueLabel == 0:
            negativeClassTrue.append(currTrueLabel)
            negativeClassPred.append(currPred)

    negativeClassTrue = np.array(negativeClassTrue)
    negativeClassPred = np.array(negativeClassPred)            
    negativeAccuracy = accuracy_score(negativeClassTrue, negativeClassPred)
    
    return negativeAccuracy

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        #predTrain = self.model.predict_proba(self.x)
        #Ytrain = self.y
        #self.auroc = auRoc(Ytrain, predTrain)
        #self.auprc = auPrc(Ytrain, predTrain)
        #self.balanced_accuracy = balancedAccuracy(Ytrain, predTrain)
        #self.sensitivity = sensitivity(Ytrain, predTrain)
        #self.specificity = specificity(Ytrain, predTrain)        
       
        
        predValid = self.model.predict_proba(self.validation_data[0])
        Yvalid = self.validation_data[1]
        self.val_auroc = auRoc(Yvalid, predValid)
        self.val_auprc = auPrc(Yvalid, predValid)
        self.val_balanced_accuracy = balancedAccuracy(Yvalid, predValid)
        self.val_sensitivity = sensitivity(Yvalid, predValid)
        self.val_specificity = specificity(Yvalid, predValid)
        print("val_auroc:", self.val_auroc, ",val_auprc:", self.val_auprc, ",val_balanced_accuracy", self.val_balanced_accuracy, ",val_sensitivity:", self.val_sensitivity, ",val_specificity:", self.val_specificity)
        return


def get_model(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length):
    model = Sequential()

    conv1_layer = Conv1D(input_shape=(length, 4),
                        padding="valid",
                        strides=1,
                        activation="relu",
                        kernel_size=8,
                        filters=1000,
                        use_bias=True)


    model.add(conv1_layer)

    for i in range(numConvLayers-1):
        convn_layer = Conv1D(padding="valid",
                        strides=1,
                        activation="relu",
                        kernel_size=8,
                        filters=numConvFilters,
                        use_bias=True)
        model.add(convn_layer)
    

    model.add(MaxPooling1D(pool_size=13, strides=13))

    model.add(Dropout(poolingDropout))

    model.add(Flatten())

    model.add(Dense(units=numLabels, use_bias=True))
    model.add(Activation('sigmoid'))
    
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
                     poolingDropout,
                     learningRate,
                     momentum,
                     length,
                     pretrainedModel,
                     positiveClassWeight,
                     negativeClassWeight
                    ):
    classWeights = {0:negativeClassWeight, 1:positiveClassWeight}
    numLabels = Y_train.shape[1]
    if pretrainedModel:
        model = load_model(pretrainedModel)
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length)

    optim = optimizers.SGD(lr=learningRate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', specificity_metric])    
    model.summary()
    
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=60, verbose=0, mode='min')
    print(X_train.shape)
    print(Y_train.shape)
    metrics = Metrics()
    model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer, earlystopper, metrics], class_weight = classWeights)

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
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.00)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=1000)
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model', required=False, default=None)
    parser.add_argument('-c1w', '--class-1-weight', type=int, help='weight for positive class during training', required=False, default=1)
    parser.add_argument('-c2w', '--class-2-weight', type=int, help='weight for positive class during training', required=False, default=1)
   
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
                     poolingDropout=args.pool_dropout_rate,
                     learningRate=args.learning_rate,
                     momentum=args.momentum,
                     length=args.length,
                     pretrainedModel=args.pretrained_model,
                     positiveClassWeight=args.class_1_weight,
                     negativeClassWeight=args.class_2_weight
                    )
