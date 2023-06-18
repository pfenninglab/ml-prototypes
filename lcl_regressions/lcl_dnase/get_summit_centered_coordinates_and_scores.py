import argparse
import pandas as pd

def generate_summit_centered_output(infile, length, outfile, chromSizesFile):
    chromSizes = dict()
    with open(chromSizesFile, 'r') as f:
        for line in f:
            [chrom, size] = line.strip().split("\t")
            chromSizes[chrom] = int(size)
            
    
    peaks_info = pd.read_csv(infile,
                             compression='gzip', 
                             names=["CHR", "START", "END", "NAME", "SCORE", "STRAND", "SIGNAL", "P", "Q", "SUMMIT"],
                             sep='\t')

    convert_dict = {'START': int, 
                     'END': int,
                     'SUMMIT': int
               } 
  
    peaks_info = peaks_info.astype(convert_dict) 

    
    #filter out peaks with no defined summits
    peaks_info = peaks_info[peaks_info.SUMMIT != -1]
    peaks_info.START = peaks_info.START + peaks_info.SUMMIT - length/2
    peaks_info.END = peaks_info.START + length
    peaks_info.SUMMIT = length/2
    

    
    peaks_info =  peaks_info[peaks_info['START']>=0]
    peaks_info = peaks_info[peaks_info['END']<=[chromSizes[chrom] for chrom in peaks_info['CHR']]]
    
    peaks_info = peaks_info.astype({'START': 'int', 'END': 'int'})

    peaks_info.to_csv(outfile,
                      compression='gzip',
                      sep='\t',
                      header=False,
                      index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get summit centered coordinates from a narrow peak file', fromfile_prefix_chars='@')
    parser.add_argument('-i', '--input', help='path to narrow peak file input', required=True)
    parser.add_argument('-l', '--length', type=int, help='length around summit desired', required=True)
    parser.add_argument('-s', '--sizes', help='path to chrom sizes file for this genome', required=True)
    parser.add_argument('-o', '--output', help='path to narrow peak file output', required=True)
    
    args = parser.parse_args()
    
    infile = args.input
    length = args.length
    outfile = args.output
    chromSizesFile = args.sizes
    
    generate_summit_centered_output(infile, length, outfile, chromSizesFile)
    