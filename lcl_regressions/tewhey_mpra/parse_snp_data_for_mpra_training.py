import pandas as pd
import numpy as np
import os



def oneHotEncodeSequence(sequence):
    oneHotDimension = (len(sequence), 4)
    dnaAlphabet = {"A":0, "G":1, "C":2, "T":3}
    one_hot_encoded_sequence = np.zeros(oneHotDimension, dtype=np.int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in dnaAlphabet:
            index = dnaAlphabet[nucleotide.upper()]
            one_hot_encoded_sequence[i][index] = 1
    return one_hot_encoded_sequence

def getNumpyArrays(split_snp_data):

    split_sequences = []
    split_labels = []
    for index, snp_row in split_snp_data.iterrows():
        refSequence = snp_row["REF.sequence.1kb"]
        refExpression = snp_row["C.A.log2FC"]
        altSequence = snp_row["ALT.sequence.1kb"]
        altExpression = snp_row["C.B.log2FC"]

        if not str(refExpression) == "nan":
            split_sequences.append(oneHotEncodeSequence(refSequence))
            split_labels.append([refExpression])        
        
        if not str(altExpression) == "nan":
            split_sequences.append(oneHotEncodeSequence(altSequence))
            split_labels.append([altExpression])    

    split_sequences = np.stack(split_sequences, axis=0)
    split_labels = np.array(split_labels)

    print(split_sequences.shape)
    print(split_labels.shape)

    return split_sequences, split_labels

def constructTrainTestValidationSets(inFile, outPrefix, trainChrs, validChrs, testChrs):
    snp_data = pd.read_csv(inFile, sep="\t")

    train_snp_data = snp_data.loc[snp_data["chr"].isin(trainChrs)]
    valid_snp_data = snp_data.loc[snp_data["chr"].isin(validChrs)]
    test_snp_data = snp_data.loc[snp_data["chr"].isin(testChrs)]


    trainSequences, trainLabels = getNumpyArrays(train_snp_data)
    validSequences, validLabels = getNumpyArrays(valid_snp_data)
    testSequences, testLabels = getNumpyArrays(test_snp_data)

    trainXOutput = "mpra_train_X.npy"
    validXOutput = "mpra_valid_X.npy"
    testXOutput = "mpra_test_X.npy"

    trainYOutput = "mpra_train_Y.npy"
    validYOutput = "mpra_valid_Y.npy"
    testYOutput = "mpra_test_Y.npy"

    np.save(os.path.join(outPrefix, trainXOutput), trainSequences)
    np.save(os.path.join(outPrefix, validXOutput), validSequences)
    np.save(os.path.join(outPrefix, testXOutput), testSequences)

    np.save(os.path.join(outPrefix,trainYOutput), trainLabels)
    np.save(os.path.join(outPrefix,validYOutput), validLabels)
    np.save(os.path.join(outPrefix,testYOutput), testLabels)


if __name__=="__main__":
    inFile = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_snps_data_extended_joined_to_1kg_mappings_with_1kb_snp_centered_sequences_with_gm_overlaps.txt"
    outPrefix = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_mpra_training/"
    trainChrs = [17, 11, 19, 12, 5, 22, 9, 21]
    validChrs = [3, 7]
    testChrs = [1, 6, 10, 4, 2, 16, 20, 15, 8, 14, 18, 13]

    constructTrainTestValidationSets(inFile, outPrefix, trainChrs, validChrs, testChrs)
