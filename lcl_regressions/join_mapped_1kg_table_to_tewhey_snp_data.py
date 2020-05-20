if __name__=="__main__":
    tewhey_snp_info_file = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_snp_data_extended.txt"
    tewhey_snp_mappings_to_1kg_file = "/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/ml_lcl/tewhey_snps_mapping_to_1kg_phase1.txt"
    
    
    tewhey_1kg_mappings = dict()
    with open(tewhey_snp_mappings_to_1kg_file, 'r') as f:
        mapping_header = f.readline().strip()
        for line in f:
            data = line.strip().split("\t")
            tewhey_id = data[0]
            tewhey_1kg_mappings[tewhey_id] = line.strip()
                
    
        
    with open(tewhey_snp_info_file, 'r') as f:
        tewhey_header = f.readline().strip()
        print("\t".join([tewhey_header, mapping_header]))
        for line in f:
            data = line.strip().split("\t")
            tewhey_id = data[1]
            if tewhey_id.startswith("MERGED"):
                continue
            mapping_info = tewhey_1kg_mappings[tewhey_id]
            print("\t".join([line.strip(),mapping_info]))
            
            