import argparse
from Bio import SeqIO

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='print sequence lengths of fasta file', fromfile_prefix_chars='@')
    parser.add_argument('-f', '--fasta', help='input fasta', required=True)
   
    args = parser.parse_args()
    fastaFile = args.fasta

    for record in SeqIO.parse(fastaFile, "fasta"):
        print len(record.seq)
    