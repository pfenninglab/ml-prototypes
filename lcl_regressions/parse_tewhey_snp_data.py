def readTewheySnpInfo(tewhey_snp_info_file):
    rsVars = dict()
    chrVars = dict()
    with open(tewhey_snp_info_file, 'r') as f:
        header = f.readline().strip().split("\t")
        for line in f:
            data = line.strip().split("\t")
            if data[1].startswith("rs"):
                rsVars[data[1]] = data
            elif data[1].startswith("chr"):
                orig_keys = data[1].split(":")
                new_keys = ":".join(orig_keys[0:2])
                chrVars[new_keys] = data
    
    return rsVars, chrVars
    
def extendSnpInfo(rsVars, chrVars, snp_database_file):
    with open(snp_database_file, 'r') as f:
        for line in f:
            data = line.strip().split("\t")
            if not (data[0] == "XY" or data[0] == "MT"):
                rsid = data[1]
                if rsid in rsVars:
                    rsVars[rsid].extend(data[0:2]+data[3:6])
                else:
                    chrom = "chr"+data[0]
                    pos = int(data[3])
                    key = ":".join([chrom, str(pos)])
                    if key in chrVars:
                        if "I" in chrVars[key][1] or "D" in chrVars[key][1]:
                            # filter out single nucleotide variants that are at the same location at indel chr variants
                            if not (len(data[4])==1 and len(data[5])==1):
                                chrVars[key].extend(data[0:2]+data[3:6])
                        else:
                            # keep single nucleotide chr variants
                            chrVars[key].extend(data[0:2]+data[3:6])
    
    return rsVars, chrVars
    
    
if __name__=="__main__":
    tewhey_snp_info_file = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_snp_data_extended.txt"
    snp_database_file = "/home/eramamur/resources/1000G_phase1_plink_files/1kg_phase1_all.bim"
    
    rsVars, chrVars = readTewheySnpInfo(tewhey_snp_info_file)
    rsVars, chrVars = extendSnpInfo(rsVars, chrVars, snp_database_file)
    
    expected_columns = 24
    
    print("\t".join(["tewhey_id", "chr", "1kg_mapped_id", "pos", "hg19_alt", "hg19_ref"]))
    for key in rsVars:
        print("\t".join([rsVars[key][1]] + rsVars[key][expected_columns-5:expected_columns]))
                                
    for key in chrVars:         
        print("\t".join([chrVars[key][1]] + chrVars[key][expected_columns-5:expected_columns]))