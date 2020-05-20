from ucscgenome import Genome
import numpy as np

def isOverlapping(targetVariantLongestStart,
                     targetVariantLongestEnd,
                     nearbyVariantLongestStart,
                     nearbyVariantLongestEnd):
    
    if targetVariantLongestStart < nearbyVariantLongestStart and targetVariantLongestEnd <= nearbyVariantLongestStart:
        return False
    if nearbyVariantLongestStart < targetVariantLongestStart and nearbyVariantLongestEnd <= targetVariantLongestStart:
        return False
    
    return True

def constructDictOfNearbyVariants(nearby_variants_mapping_file):
    nearbyVariants = dict()
    with open(nearby_variants_mapping_file, 'r') as f:
        for line in f:
            data = line.strip().split("\t")
            targetVariantChr = data[0]
            targetVariantLongestStart = int(data[1])
            targetVariantLongestEnd = int(data[2])
            targetVariantID = data[3]
            nearbyVariantChr = data[4]
            nearbyVariantLongestStart = int(data[5])
            nearbyVariantLongestEnd = int(data[6])
            nearbyVariantID = data[7]
            nearbyVariantAltAllele = data[8]
            nearbyVariantRefAllele = data[9]
            nearbyVariantRefStart = nearbyVariantLongestStart
            nearbyVariantRefEnd = nearbyVariantLongestStart + len(nearbyVariantRefAllele)
            
            
            if not isOverlapping(targetVariantLongestStart,
                                 targetVariantLongestEnd,
                                 nearbyVariantLongestStart,
                                 nearbyVariantLongestEnd):
                nearbyVariantAltStart = nearbyVariantLongestStart
                nearbyVariantAltEnd = nearbyVariantAltStart + len(nearbyVariantAltAllele)
                nearbyVariantInfo = [nearbyVariantID,
                                     nearbyVariantChr,
                                     nearbyVariantAltStart,
                                     nearbyVariantAltEnd,
                                     nearbyVariantRefStart,
                                     nearbyVariantRefEnd,
                                     nearbyVariantAltAllele,
                                     nearbyVariantRefAllele,
                                     nearbyVariantLongestStart,
                                     nearbyVariantLongestEnd]

                if targetVariantID not in nearbyVariants:
                    nearbyVariants[targetVariantID] = dict()
                
                if nearbyVariantID not in nearbyVariants[targetVariantID]:
                    removableVariantsList = []
                    add = True
                    
                    for otherNearbyVariantID in nearbyVariants[targetVariantID]:
                        otherNearbyVariantLongestStart = nearbyVariants[targetVariantID][otherNearbyVariantID][8]
                        otherNearbyVariantLongestEnd  =  nearbyVariants[targetVariantID][otherNearbyVariantID][9]                      
                        if isOverlapping(nearbyVariantLongestStart,
                                         nearbyVariantLongestEnd,
                                         otherNearbyVariantLongestStart,
                                         otherNearbyVariantLongestEnd):
                            if nearbyVariantLongestEnd - nearbyVariantLongestStart < otherNearbyVariantLongestEnd - otherNearbyVariantLongestStart:
                                removableVariantsList.append(otherNearbyVariantID)
                            else:
                                add = False
                    
                    if add:    
                        nearbyVariants[targetVariantID][nearbyVariantID] = nearbyVariantInfo
                     
                    for otherNearbyVariantID in removableVariantsList:
                        nearbyVariants[targetVariantID].pop(otherNearbyVariantID) 
                    
    return nearbyVariants

def getUpdatedPaddings(alleleLength, l, r, editR=True):
    for i in range(alleleLength-1):
        if editR:
            r = r-1
            editR = False
        else:
            l = l-1
            editR = True    
    
    return l, r
    
def constructRefSequences(targetVariantID, data, genomeObject, l, r):
    refAllele = data[24]
    altAllele = data[23]
    chrom = "chr" + data[20]
    position = int(data[22])
    refAlleleLength = len(refAllele)
    altAlleleLength = len(altAllele)
    
    refL, refR  = getUpdatedPaddings(refAlleleLength, l, r, True)
    altL, altR  = getUpdatedPaddings(altAlleleLength, l, r, True)
    
    refSequence = genomeObject[chrom][position-1-refL:position-1]+refAllele+genomeObject[chrom][position:position+refR]
    altSequence = genomeObject[chrom][position-1-altL:position-1]+altAllele+genomeObject[chrom][position:position+altR]

    return refSequence, altSequence



def constructLeftAltSequence(leftVariants, targetPosition, l, genomeObject, chrom):
    if len(leftVariants) == 0:
        return genomeObject[chrom][targetPosition-1-l:targetPosition-1]
    genomeChunkLeft = leftVariants[0][5]
    genomeChunkRight = targetPosition-1
    sequence = ""
    for i in range(len(leftVariants)-1):
        sequence = genomeObject[chrom][genomeChunkLeft:genomeChunkRight] + sequence
        sequence = leftVariants[i][6] + sequence
        genomeChunkLeft = leftVariants[i+1][5]
        genomeChunkRight = leftVariants[i][4]
    
    
    sequence = genomeObject[chrom][genomeChunkLeft:genomeChunkRight] + sequence
    sequence = leftVariants[len(leftVariants)-1][6] + sequence
    if len(sequence) < l:
        genomeChunkRight = leftVariants[len(leftVariants)-1][4]
        remainingLength = l - len(sequence) 
        genomeChunkLeft = genomeChunkRight - remainingLength
        sequence = genomeObject[chrom][genomeChunkLeft:genomeChunkRight] + sequence
    elif len(sequence) > l:
        sequenceLength = len(sequence)
        sequence = sequence[sequenceLength-l:sequenceLength]
    
    return sequence
    
def constructRightAltSequence(rightVariants, targetPosition, targetRefAlleleLength, r, genomeObject, chrom):
    if len(rightVariants) == 0:
        start = targetPosition-1+targetRefAlleleLength
        return genomeObject[chrom][start:start+r]
    genomeChunkLeft = targetPosition - 1 + targetRefAlleleLength
    genomeChunkRight = rightVariants[0][4]
    sequence = ""
    for i in range(len(rightVariants)-1):
        sequence = sequence + genomeObject[chrom][genomeChunkLeft:genomeChunkRight]
        sequence = sequence + rightVariants[i][6]
        genomeChunkLeft = rightVariants[i][5]
        genomeChunkRight = rightVariants[i+1][4]
    
    sequence = sequence + genomeObject[chrom][genomeChunkLeft:genomeChunkRight]
    sequence = sequence + rightVariants[len(rightVariants)-1][6]


    if len(sequence) < r:
        genomeChunkLeft = rightVariants[len(rightVariants)-1][5]  
        remainingLength = r - len(sequence) 
        genomeChunkRight = genomeChunkLeft + remainingLength
        sequence = sequence + genomeObject[chrom][genomeChunkLeft:genomeChunkRight]
    elif len(sequence) > r:
        sequence = sequence[0:r]
    
    return sequence

def constructAltSequences(targetVariantID, data, nearbyVariantsToTarget, genomeObject, l, r):
    targetRefAllele = data[24]
    targetAltAllele = data[23]
    chrom = "chr" + data[20]
    targetPosition = int(data[22])
    
    targetRefAlleleLength = len(targetRefAllele)
    targetAltAlleleLength = len(targetAltAllele)

    refL, refR  = getUpdatedPaddings(targetRefAlleleLength, l, r, True)
    altL, altR  = getUpdatedPaddings(targetAltAlleleLength, l, r, True)        
    
    leftVariants = []
    rightVariants = []
    for variant in nearbyVariantsToTarget:
        nearbyVariantRefStart = nearbyVariantsToTarget[variant][4]
        
        if nearbyVariantRefStart < targetPosition-1:
            leftVariants.append(nearbyVariantsToTarget[variant])
        else:
            rightVariants.append(nearbyVariantsToTarget[variant])
    
    leftVariants = sorted(leftVariants, key=lambda x: x[4], reverse=True)
    rightVariants = sorted(rightVariants, key=lambda x: x[4], reverse=False)
    
    refLeftSequence = constructLeftAltSequence(leftVariants, targetPosition, refL, genomeObject, chrom)
    refRightSequence = constructRightAltSequence(rightVariants, targetPosition, targetRefAlleleLength, refR, genomeObject, chrom)
    
    refSequence = refLeftSequence + targetRefAllele + refRightSequence

    altLeftSequence = constructLeftAltSequence(leftVariants, targetPosition, altL, genomeObject, chrom)
    altRightSequence = constructRightAltSequence(rightVariants, targetPosition, targetRefAlleleLength, altR, genomeObject, chrom)
    
    altSequence = altLeftSequence + targetAltAllele + altRightSequence
    
    return refSequence, altSequence

def revcomp(inseq):
    mapDict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    outseq = ""
    for nuc in inseq:
        nucUpper = nuc.upper()
        outseq = mapDict[nucUpper] + outseq
    return outseq

def oneHotEncodeSequence(sequence):
    oneHotDimension = (len(sequence), 4)
    dnaAlphabet = {"A":0, "G":1, "C":2, "T":3}
    one_hot_encoded_sequence = np.zeros(oneHotDimension, dtype=np.int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in dnaAlphabet:
            index = dnaAlphabet[nucleotide.upper()]
            one_hot_encoded_sequence[i][index] = 1
    return one_hot_encoded_sequence

if __name__=="__main__":
    tewhey_variants_extended_joined_info_file = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_snps_data_extended_joined_to_1kg_mappings.txt"
    nearby_variants_mapping_file = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_snp_data_extended_joined_to_1kg_mappings_snps_within_1000bp_esv_removed.txt"
    refXOutput = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_one_hot_encoded_ref_sequences.npy"
    altXOutput = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_one_hot_encoded_alt_sequences.npy"
   
    
    genomeName = "hg19"
    genomeDir = "/home/eramamur/resources/genomes/hg19/"
    l = 499
    r = 500
    genomeObject = Genome(genomeName, cache_dir=genomeDir, use_web=False)
    
    nearbyVariants = constructDictOfNearbyVariants(nearby_variants_mapping_file)
    encodedRefSequences = []
    encodedAltSequences = []
    
    with open(tewhey_variants_extended_joined_info_file, 'r') as f:
        header = f.readline().strip()
        print "\t".join([header, "REF.sequence.1kb", "ALT.sequence.1kb"])
        for i, line in enumerate(f):
            data = line.strip().split("\t")
            targetVariantID = data[21]
            if targetVariantID == ".":
                targetVariantID = data[1]
            if targetVariantID in nearbyVariants:
                nearbyVariantsToTarget = nearbyVariants[targetVariantID]
            else:
                nearbyVariantsToTarget = None

            revOrNot = data[2]
            refOrAlt = data[3]
            if refOrAlt == "ref":
                refSequence, altSequence = constructRefSequences(targetVariantID, data, genomeObject, l, r)
            elif refOrAlt == "alt":
                refSequence, altSequence = constructAltSequences(targetVariantID, data, nearbyVariantsToTarget, genomeObject, l, r)            
            if revOrNot == "neg":
                refSequence, altSequence = revcomp(refSequence), revcomp(altSequence)
            
            print "\t".join([line.strip(), refSequence, altSequence])    
            encodedRefSequences.append(oneHotEncodeSequence(refSequence))
            encodedAltSequences.append(oneHotEncodeSequence(altSequence))

    encodedRefSequences = np.stack(encodedRefSequences, axis=0)
    encodedAltSequences = np.stack(encodedAltSequences, axis=0)
    np.save(refXOutput, encodedRefSequences)
    np.save(altXOutput, encodedAltSequences)
    
                