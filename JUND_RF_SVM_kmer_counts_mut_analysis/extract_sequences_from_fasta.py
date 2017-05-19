import numpy as np
import sys
from Bio import SeqIO

if __name__=="__main__":
	idsFile = sys.argv[1]
	seqFile = sys.argv[2]
	outFile = sys.argv[3]
	with open(idsFile) as f:
		ids = set([line.strip() for line in f])
		
	seq_subset = []
	with open(seqFile, "rU") as handle:
		for record in SeqIO.parse(handle, "fasta"):
			if record.id in ids:
				seq_subset.append(record)
	
	SeqIO.write(seq_subset, outFile, "fasta")
