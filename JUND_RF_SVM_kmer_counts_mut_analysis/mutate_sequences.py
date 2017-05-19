import numpy as np
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import matplotlib.pyplot as plt

import copy

if __name__=="__main__":
	np.random.seed(23)
	input_fasta = sys.argv[1]
	mut_freq = float(sys.argv[2])
	out_file = sys.argv[3]
	
	mut_records = []
	pos_hist = dict()
	with open(input_fasta, "rU") as handle:
		for record in SeqIO.parse(handle, "fasta"):
			seq_length = len(record.seq)
			num_mut = int(np.ceil(1.0*mut_freq*seq_length))			
			mut_positions = np.random.randint(1, high=seq_length-3, size=num_mut)
			
			for pos in xrange(seq_length):
				if pos not in pos_hist:
					pos_hist[pos] = 0
					
			for pos in mut_positions:
				pos_hist[pos] = pos_hist[pos] + 1
				mut_record = copy.deepcopy(record)
				orig_seq = mut_record.seq				
				orig_nuc = orig_seq[pos]
				possible_mut = [nuc for nuc in ["A","T","G","C"] if not nuc==orig_nuc]
				mut_nuc = np.random.choice(possible_mut)
				mut_seq = orig_seq[0:pos] + mut_nuc + orig_seq[pos+1:]
				mut_record.seq = mut_seq
				mut_record.id = "|".join([mut_record.id, str(pos+1), orig_nuc, mut_nuc])
				mut_records.append(mut_record)
				
	
	positions = []
	num_mut_at_pos = []
	
	for pos in pos_hist:
		positions.append(pos)
		num_mut_at_pos.append(pos_hist[pos])
	
	SeqIO.write(mut_records, out_file, "fasta")
	plt.scatter(positions, num_mut_at_pos)
	plt.show()
	plt.savefig('pos_hist.png')
