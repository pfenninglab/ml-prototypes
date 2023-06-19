# Training and evaluation of 

This folder contains code to train and evaluate a CNN model on signal values from narrowPeak files of GM12878 lymphoblastoid cell line (LCL) DNase-seq data

## Training a CNN regression model on LCL DNase-seq

1. I obtained 1000bp coordinates surrounding the summits of LCL DNase-seq IDR peaks.
```
python get_summit_centered_coordinates_and_scores.py -i data/lcl/	lcl_human_idr_conservative_peak.narrowPeak.gz -l 1000 -o data/lcl/lcl_human_idr.conservative_peak.summit_centered_l1000.narrowPeak.gz -s hg38.chrom.sizes

zcat data/lcl/lcl_human_idr.conservative_peak.summit_centered_l1000.narrowPeak.gz | sort -k1,1 -k2,2n > data/lcl/lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak
```

2. I split the resulting file by chromosome to derive a training, test, and validation set for model training
```
grep 'chr4' data/lcl/lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak > data/lcl/chr4_lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak
grep 'chr8\|chr9' data/lcl/lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak > data/lcl/chr8_9_lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak
sed '/chr4\|chr8\|chr9/d' data/lcl/lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak > data/lcl/chrs_training_lcl_human_idr.conservative_peak.summit_centered_l1000.sorted.narrowPeak
```

3. I extracted one-hot encoded numpy arrays for the sequences in the training set, validation set, and test set
```
python get_one_hot_encoded_sequences.py -i data/lcl/chr4_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.narrowPeak -xo data/lcl/chr4_summit_centered_validation_set_hg19_X.npy -g hg19 -r /path/to/hg19/
python get_one_hot_encoded_sequences.py -i data/lcl/chr8_9_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.narrowPeak -xo data/lcl/chr8_9_summit_centered_test_set_hg19_X.npy -g hg19 -r /path/to/hg19/
python get_one_hot_encoded_sequences.py -i data/lcl/chrs_training_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.narrowPeak -xo data/lcl/chrs_training_summit_centered_training_set_hg19_X.npy -g hg19 -r /path/to/hg19/
```
5. I extracted bed files for input to a script that requires 5 column bed files
```
awk -vFS='\t' -vOFS='\t' '{print $1,$2,$3,".",$7}' data/lcl/chr4_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.narrowPeak > data/lcl/chr4_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.bed
awk -vFS='\t' -vOFS='\t' '{print $1,$2,$3,".",$7}' data/lcl/chr8_9_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.narrowPeak > data/lcl/chr8_9_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.bed
awk -vFS='\t' -vOFS='\t' '{print $1,$2,$3,".",$7}' data/lcl/chrs_training_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.sorted.narrowPeak > data/lcl/chrs_training_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.bed
```
7. I extracted numpy arrays for the signal values which will be the output of the model
```
python construct_numpy_arrays.py -i data/lcl/chr4_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.bed -yo data/lcl/chr4_summit_centered_validation_set_hg19_Y.npy
python construct_numpy_arrays.py -i data/lcl/chr8_9_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.bed -yo data/lcl/chr8_9_summit_centered_test_set_hg19_Y.npy
python construct_numpy_arrays.py -i data/lcl/chrs_training_lcl_human_idr.hg19.conservative_peak.summit_centered_l1000.bed -yo data/lcl/chrs_training_summit_centered_training_set_hg19_Y.npy
```
8. 
