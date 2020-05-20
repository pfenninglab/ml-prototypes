library(gkmSVM)
library(GenomicRanges)
library(Matrix)
library(rtracklayer)
library(BiocGenerics)
library(Biostrings)
library(GenomeInfoDb)
library(IRanges)
library(S4Vectors)

posBedFN = '/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.bed'
  
# output file names:  
posfn= '/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.fa'   #positive set (FASTA format)
negfn= '/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.genNullSeq.orig.negatives.fa'   #negative set (FASTA format)

genNullSeqs(posBedFN, genomeVersion = "hg19",
	outputPosFastaFN = posfn, outputNegFastaFN = negfn);