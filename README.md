# Machine Learning scripts

This is a place to put all rough scripts and prototypes of scripts that are yet to be converted to final product. These may be tailored to the specific task and also may be quick scripts put together just to run a machine learning algorithm on a dataset.

## Kmer Count Converter
Function to convert a given set of sequences into a k-mer count representation. Can also specify wildcard and number of wildcard characters in k-mers.

```
python kmerizer.py [-h] -i FASTA -k KMER_LENGTH -m MAX_CONSEC_WC
                   [-v KMER_VOCAB_FILE] -o DESIGN_MATRIX_OUT
                   [-vo KMER_VOCAB_OUT]
```

Arguments description:

```
  -h, --help            show this help message and exit
  -i FASTA, --fasta FASTA
                        fasta file input
  -k KMER_LENGTH, --kmer-length KMER_LENGTH
                        length of k-mer desired
  -m MAX_CONSEC_WC, --max-consec-wc MAX_CONSEC_WC
                        maximum continuous wildcards
  -v KMER_VOCAB_FILE, --kmer-vocab-file KMER_VOCAB_FILE
                        file containing kmer vocabulary
  -o DESIGN_MATRIX_OUT, --design-matrix-out DESIGN_MATRIX_OUT
                        file to output design matrix in
  -vo KMER_VOCAB_OUT, --kmer-vocab-out KMER_VOCAB_OUT
                        file to output kmer vocabulary
```
