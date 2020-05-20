714/101: classif_Ytest.shape
714/102: classif_Xvalid.shape
714/103: classif_Yvalid.shape
   1: ls
   2: import padnas as pd
   3: import pandas as pd
   4: rep1_rep2_data = pd.read_csv("rep1_rep2.idr0.05.unthresholded-peaks.txt.gz", sep="\t", compression="gzip")
   5: rep1_rep2_data.head
   6: rep1_rep2_data.head()
   7:
rep1_rep2_data = pd.read_csv("rep1_rep2.idr0.05.unthresholded-peaks.txt.gz"
                            ,sep="\t", compression="gzip"
                            , names=["CHR",
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
   8: rep1_rep2_data
   9: rep1_rep2_data.head()
  10: import matplotlibpyplot as plt
  11: import matplotlib.pyplot as plt
  12: import numpy as np
  13: plt.scatter(np.log(rep1_rep2_data.REP1_SIGNAL), np.log(rep1_rep2_data.REP2_SIGNAL))
  14: plt.show()
  15: plt.close()
  16: plt.scatter(np.log(rep1_rep2_data.REP1_SIGNAL), np.log(rep1_rep2_data.REP2_SIGNAL))
  17: plt.savefig("/home/eramamur/test_rep1_rep2_gm.png")
  18: plt.close()
  19: plt.scatter(np.log10(rep1_rep2_data.REP1_SIGNAL), np.log10(rep1_rep2_data.REP2_SIGNAL))
  20: plt.savefig("/home/eramamur/test_rep1_rep2_gm.png")
  21: plt.scatter(rep1_rep2_data.REP1_SIGNAL, rep1_rep2_data.REP2_SIGNAL)
  22: plt.savefig("/home/eramamur/test_rep1_rep2_gm.png")
  23: plt.scatter(np.log10(rep1_rep2_data.REP1_SIGNAL), np.log10(rep1_rep2_data.REP2_SIGNAL))
  24: plt.savefig("/home/eramamur/test_rep1_rep2_gm_log10.png")
  25: plt.close()
  26: plt.scatter(np.log10(rep1_rep2_data.REP1_SIGNAL), np.log10(rep1_rep2_data.REP2_SIGNAL))
  27: plt.savefig("/home/eramamur/test_rep1_rep2_gm_log10.png")
  28: plt.close()
  29: colors = ['k' for _ in range(rep1_rep2_data.shape[0])]
  30: colors[rep1_rep2_data["GLOBALIDR] > 0.05] = 'r'
  31: colors[rep1_rep2_data["GLOBALIDR"] > 0.05] = 'r'
  32: rep1_rep2_data["GLOBALIDR"]>0.05
  33: colors[list(rep1_rep2_data["GLOBALIDR"] > 0.05)] = 'r'
  34: colors = numpy.zeros(rep1_rep2_data.shape[0], dtype=str)
  35: colors = np.zeros(rep1_rep2_data.shape[0], dtype=str)
  36: colors[:]='k'
  37: colors[list(rep1_rep2_data["GLOBALIDR"] > 0.05)] = 'r'
  38: plt.scatter(np.log10(rep1_rep2_data.REP1_SIGNAL), np.log10(rep1_rep2_data.REP2_SIGNAL), edgecolor=colors, c=colors, alpha=0.05)
  39: plt.savefig("/home/eramamur/test_rep1_rep2_gm_log10.png")
  40: plt.close()
  41: plt.scatter(np.log10(rep1_rep2_data.REP1_SIGNAL), np.log10(rep1_rep2_data.REP2_SIGNAL), edgecolor=colors, c=colors, alpha=0.05)
  42: plt.savefig("/home/eramamur/test_rep1_rep2_gm_log10.png")
  43: plt.close()
  44: colors
  45: colors=='k'
  46: np.sum(colors=='k')
  47:
np.sum(colors=='r'_
)
  48: np.sum(rep1_rep2_data["GLOBALIDR"] > 0.05)
  49: rep1_rep2_dat.ashape
  50: rep1_rep2_data.shape
  51: rep1_rep2_data.head()
  52: np.sum(rep1_rep2_data["GLOBALIDR"] < -np.log10(0.05))
  53: colors = np.zeros(rep1_rep2_data.shape[0], dtype=str)
  54: colors[:]='k'
  55: colors[list(rep1_rep2_data["GLOBALIDR"] < -np.log10(0.05))] = 'r'
  56: plt.scatter(np.log10(rep1_rep2_data.REP1_SIGNAL), np.log10(rep1_rep2_data.REP2_SIGNAL), edgecolor=colors, c=colors, alpha=0.05)
  57: plt.savefig("/home/eramamur/test_rep1_rep2_gm_log10.png")
  58: plt.close()
  59: clear
  60: rep1_rep2_data.head()
  61: from scipy.stats import pearsonr
  62: rep1_rep2_data_filtered = rep1_rep2_data.loc[rep1_rep2_data["GLOBALID"] < -np.log10(0.05)]
  63: rep1_rep2_data_filtered = rep1_rep2_data.loc[rep1_rep2_data["GLOBALIDR"] < -np.log10(0.05)]
  64: rep1_rep2_data_filtered.shape
  65: pearsonr(rep1_rep2_data_filtered["REP1_SIGNAL"], rep1_rep2_data_filtered["REP2_SIGNAL"])
  66: rep1_rep2_data_filtered = rep1_rep2_data.loc[rep1_rep2_data["LOCALIDR"] < -np.log10(0.05)]
  67: rep1_rep2_data_filtered.shape
  68: pearsonr(rep1_rep2_data_filtered["REP1_SIGNAL"], rep1_rep2_data_filtered["REP2_SIGNAL"])
  69: rep1_rep2_data_filtered = rep1_rep2_data.loc[rep1_rep2_data["GLOBALIDR"] > -np.log10(0.05)]
  70: rep1_rep2_data_filtered.shape
  71: rep1_rep2_data_filtered = rep1_rep2_data.loc[rep1_rep2_data["LOCALIDR"] > -np.log10(0.05)]
  72: rep1_rep2_data_filtered.shape
  73: rep1_rep2_data_filtered = rep1_rep2_data.loc[rep1_rep2_data["GLOBALIDR"] > -np.log10(0.05)]
  74: rep1_rep2_data_filtered.shape
  75: pearsonr(rep1_rep2_data_filtered["REP1_SIGNAL"], rep1_rep2_data_filtered["REP2_SIGNAL"])
  76: history
  77: history -g -f check_replicate_correlations.py
