# Machine Learning scripts

This is a place to put all rough scripts and prototypes of scripts that are yet to be converted to final product. These may be tailored to the specific task and also may be quick scripts put together just to run a machine learning algorithm on a dataset.

## JUND Analysis Scripts

* Script to convert input sequences to k-mer count representations
* Script to simulate random substitutions on input sequences
* Script to extract a subset of sequences from a fasta file
* Ipython Notebook to run a random forest and SVM classifier on k-mer representations of JUND ChIP-Seq peaks

## Jemmie 3expHG DanQ Like Model

* Ipython Notebook that contains all code including preprocessing and model training/testing on cell type specific H3K27ac peaks from the 5 control samples
* Dependency on ucscgenome, intervaltree, pandas
* Extracts peak data from narrowPeak bed files
* For each bin (currently 200 bp) in genome, calculate binary label based on condition whether bin lies at least half in peaks (uses interval tree)
* Train a DanQ-like model
