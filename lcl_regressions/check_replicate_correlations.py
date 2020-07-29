import pandas as pd
import argparse
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['svg.fonttype'] = 'none'
rcParams['font.size']=15


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='get correlations between replicates', fromfile_prefix_chars='@')
    parser.add_argument('-i', '--input', help='input text file (bedtools subtract of klab unthresholded output and blacklist)', required=True)
    args = parser.parse_args()
    inFile = args.input

    rep1_rep2_data = pd.read_csv(inFile,
                                 sep="\t", compression="gzip",
                                 names=["CHR",
                                 "START",
                                 "END",
                                 "NAME",
                                 "SCORE",
                                 "STRAND",
                                 "SIGNAL",
                                 "P",
                                 "Q",
                                 "SUMMIT",
                                 "LOCALIDR",
                                 "GLOBALIDR",
                                 "REP1_START",
                                 "REP1_END",
                                 "REP1_SIGNAL",
                                 "REP1_SUMMIT",
                                 "REP2_START",
                                 "REP2_END",
                                 "REP2_SIGNAL",
                                 "REP2_SUMMIT"]
                                )

    print("Overall correlations")
    print(rep1_rep2_data.shape)
    print(pearsonr(rep1_rep2_data["REP1_SIGNAL"], rep1_rep2_data["REP2_SIGNAL"]))
    print(spearmanr(rep1_rep2_data["REP1_SIGNAL"], rep1_rep2_data["REP2_SIGNAL"]))

    chr4_rep1_rep2_data = rep1_rep2_data.loc[rep1_rep2_data["CHR"]=="chr4"]
    print("chr4 correlations")
    print(chr4_rep1_rep2_data.shape)
    print(pearsonr(chr4_rep1_rep2_data["REP1_SIGNAL"], chr4_rep1_rep2_data["REP2_SIGNAL"]))
    print(spearmanr(chr4_rep1_rep2_data["REP1_SIGNAL"], chr4_rep1_rep2_data["REP2_SIGNAL"]))

    chrs_training_rep1_rep2_data = rep1_rep2_data.loc[~rep1_rep2_data["CHR"].isin(["chr4", "chr8", "chr9"])]
    print("Training chr correlations")
    print(chrs_training_rep1_rep2_data.shape)
    print(pearsonr(chrs_training_rep1_rep2_data["REP1_SIGNAL"], chrs_training_rep1_rep2_data["REP2_SIGNAL"]))
    print(spearmanr(chrs_training_rep1_rep2_data["REP1_SIGNAL"], chrs_training_rep1_rep2_data["REP2_SIGNAL"]))

    plt.scatter(chr4_rep1_rep2_data["REP1_SIGNAL"], chr4_rep1_rep2_data["REP2_SIGNAL"], s=5, alpha=0.5)
    plt.ylabel("DNase signal (GM12878 rep1)")
    plt.xlabel("DNase signal (GM12878 rep2)")
    plt.savefig("gm_replicate_concordance.svg")
    plt.close()