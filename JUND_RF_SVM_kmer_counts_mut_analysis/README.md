# Random Forest and SVM classifier on k-mer features of JunD peaks

## Random Forest and SVM classifier
* IPython notebook to train random forest and SVM models using sklearn on k-mer count features of peaks vs flanks
* 50:50 split of training and test sets with flanks as negative set
* Assumes peaks and flanks are provided in a tsv file with peaks on the top rows followed by flanks below them
* Random forest and SVM (RBF kernel) both use default params

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

## Subsetting a FASTA file
Function to retrieve a subset of sequences from a fasta file given a set of ids

```
python extract_sequences_from_fasta.py ids_file input_fasta output_fasta
```

Arguments description:

```
  ids_file              ids to be extracted from the fasta input file
  input_fasta           fasta file containing input set of sequences
  output_fasta           output fasta file that contains only the subset of sequences that are present in the ids_file  
```

