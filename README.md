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

## Sequence Mutator
Function to perform mutations in input sequences such that each output sequence each contains one substitution as compared to the reference input sequence.
```
python mutate_sequences.py input_fasta mutation_freq output_file
```

Arguments description:

```
  input_fasta           reference sequences to be mutated
  mutation_freq         mutation frequency per nucleotide (if 2.0 then there will be 2 mutations per nucleotide in the input sequences)
  output_file           output fasta file containing mutated sequences (fasta sequence headers contain info on reference sequence, position and type of substitution made  
```



