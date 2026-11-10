root_path = "/home/b_lin/Documents/Coding/nanopore/faux_data_RNN_seq/"
train_dataset_path = root_path + "data/train/"
test_dataset_path = root_path + "data/test/"

sequence_path = root_path + "data_generation/several_genes.txt"
kmer_map_path = root_path + "data_generation/9mer_levels_v1.txt"

k = 9
stride = 1
c1_kernel_size = k
c1_pad = 1

DEBUG = False