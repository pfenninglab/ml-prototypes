from Bio import SeqIO
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
import numpy as np
import sys
import argparse


def generate_wildcard_kmers(kmer, max_consec_wildcard):
    if 0<max_consec_wildcard and max_consec_wildcard>len(kmer):
        sys.exit("max_consec_wildcard not in range 0 to k")
    wc_kmers = []
    
    for num_consec in xrange(1,max_consec_wildcard+1):
        for i in xrange(len(kmer)-num_consec+1):
            new_kmer = kmer[0:i] + "N"*num_consec + kmer[i+num_consec:]
            wc_kmers.append(new_kmer)
            
    return wc_kmers


def kmerize_fa(input_fasta, k, max_consec_wildcard, design_matrix_out, kmer_vocab):
    outf = open(design_matrix_out, 'w')
    header = ["seq_id"] + kmer_vocab
    outf.write("\t".join(header))
    outf.write("\n")
    
    for record in SeqIO.parse(input_fasta, 'fasta'):
        curr_record_kmer_counts = dict()
        for kmer in kmer_vocab:
            curr_record_kmer_counts[kmer] = 0
        seq = record.seq
        for i in xrange(len(seq)-k+1):
            kmer = seq[i:i+k]
            revcompkmer = kmer.reverse_complement()
            kmer = str(kmer).upper()
            revcompkmer = str(revcompkmer).upper()
            
            if kmer in curr_record_kmer_counts:
                curr_record_kmer_counts[kmer] =  curr_record_kmer_counts[kmer] + 1
            elif revcompkmer in curr_record_kmer_counts:
                curr_record_kmer_counts[revcompkmer] = curr_record_kmer_counts[revcompkmer] + 1
            
            for wc_kmer in generate_wildcard_kmers(kmer, max_consec_wildcard):
                revcompwckmer = str(Seq(wc_kmer, generic_dna).reverse_complement()).upper()
                if wc_kmer in curr_record_kmer_counts:
                    curr_record_kmer_counts[wc_kmer] = curr_record_kmer_counts[wc_kmer] + 1
                elif revcompwckmer in curr_record_kmer_counts:
                    curr_record_kmer_counts[revcompwckmer] = curr_record_kmer_counts[revcompwckmer] + 1
                    
        curr_record_counts = []
        for kmer in kmer_vocab:
            count = curr_record_kmer_counts[kmer]
            curr_record_counts.append(count)
        line = [record.id] + [str(val) for val in curr_record_counts]
        outf.write("\t".join(line))
        outf.write("\n")
        
    outf.close()
    return


def get_kmer_vocab(input_fasta, k, max_consec_wildcard):
    kmer_vocab = set()
    for record in SeqIO.parse(input_fasta, 'fasta'):
        sequence = record.seq
        for i in xrange(len(sequence)-k+1):
            kmer = sequence[i:i+k]
            revcompkmer = kmer.reverse_complement()
            
            kmer = str(kmer).upper()
            revcompkmer = str(revcompkmer).upper()
            
            if kmer not in kmer_vocab and revcompkmer not in kmer_vocab:
                kmer_vocab.add(kmer)
            
            for wc_kmer in generate_wildcard_kmers(kmer, max_consec_wildcard):
                revcompwckmer = str(Seq(wc_kmer, generic_dna).reverse_complement()).upper()
                if wc_kmer not in kmer_vocab and revcompwckmer not in kmer_vocab:
                    kmer_vocab.add(wc_kmer)
    return list(kmer_vocab)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Kmerizer for Fasta file')
    parser.add_argument('-i', '--fasta', help='fasta file input', required=True)
    parser.add_argument('-k', '--kmer-length', type=int, help='length of k-mer desired', required=True)
    parser.add_argument('-m', '--max-consec-wc', type=int, help='maximum continuous wildcards', required=True)
    parser.add_argument('-v', '--kmer-vocab-file', help='file containing kmer vocabulary', required=False)
    parser.add_argument('-o', '--design-matrix-out', help='file to output design matrix in', required=True)
    parser.add_argument('-vo', '--kmer-vocab-out', help='file to output kmer vocabulary', required=False)
    
    args = parser.parse_args()
    
    print "Getting kmer vocab"
    kmer_vocab = []
    if args.kmer_vocab_file is None:
        kmer_vocab = get_kmer_vocab(args.fasta, args.kmer_length, args.max_consec_wc)
    else:
        with open(args.kmer_vocab_file, 'r') as f:
            for line in f:
                kmer_vocab.append(line.strip())
    
    print "Writing kmer vocab"
    outf = open(args.kmer_vocab_out, 'w')
    for kmer in kmer_vocab:
        outf.write(kmer+"\n")
    outf.close()
    
    print "Getting kmer counts"
    kmerize_fa(args.fasta, args.kmer_length, args.max_consec_wc, args.design_matrix_out, kmer_vocab)