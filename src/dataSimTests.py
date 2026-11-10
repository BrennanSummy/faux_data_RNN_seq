import constants
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader

#vvvvvvvvvvvvvvvvvvvv NAMING CONVENTION vvvvvvvvvvvvvvvvvvvv
# Functions are named with: oneTwoThree
# Classes   are named with: OneTwoThree
# Variables are named with: one_two_three
#^^^^^^^^^^^^^^^^^^^^ NAMING CONVENTION ^^^^^^^^^^^^^^^^^^^^

#vvvvvvvvvvvvvvvvvvvv . vvvvvvvvvvvvvvvvvvvv
#^^^^^^^^^^^^^^^^^^^^ . ^^^^^^^^^^^^^^^^^^^^



class KmerHandler():
    '''Handles Kmer map files, creating and storing appropriate dictionaries'''
    def __init__(self, map_filepath,k):
        self.ACGT_to_idx_dict = {"A":1,"C":2,"G":3,"T":4}
        self.idx_to_ACGT_dict = {val:key for key,val in self.ACGT_to_idx_dict.items()}
        # Read the map file and create a dictionary for kmer string keys and corresponding Z scores
        self.k = k
        self.kmer_map = {}
        with open(map_filepath,'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                kmer_string, z = self.readKmerMapLine(line)
                self.kmer_map[kmer_string] = z
        self.index_kmer_dict = {i:kmer for i,kmer in enumerate(self.kmer_map.keys())}
        self.kmer_index_dict = {kmer:i for i,kmer in self.index_kmer_dict.items()}



    def readKmerMapLine(self,line_string):
        kmer_string = line_string[0:self.k]

        z = line_string[self.k+1:-1]

        z = float(z)

        return kmer_string, z

    def printMap(self,n0,n1):
        '''
        Print the items of the kmer to Z score dictionary in the specified range
        Args:
            n0: The first item to print
            n1: The last item to print
        '''
        print(list(self.kmer_map.items())[n0:n1])

    def kmerIndexSeqToBaseSeq(self,sequence_of_kmer_indices : list[int]) -> str:
        base_list = [self.index_kmer_dict[index][0] for index in sequence_of_kmer_indices]
        last_kmer_index = sequence_of_kmer_indices[len(sequence_of_kmer_indices)-1]
        base_list += [self.index_kmer_dict[last_kmer_index][1:]]
        sequence = ''.join(base_list)
        idx_sequence = [self.ACGT_idx_dict[base] for base in sequence]
        return sequence,idx_sequence
    
class FauxDataGenerator():
    '''Given a kmer map, stores it, with the ability to generate faux data given a sequence of kmer indices as input'''
    def __init__(self, k:int, kmer_handler:KmerHandler, global_mean, global_std_dev,dwell_time_mean, dwell_time_std_dev):
        '''ASSUMES A 5' TO 3' SEQUENCE'''
        self.k = k
        self.kmer_handler = kmer_handler
        self.kmer_index_expected_zcurrent_dict = {kmer_handler.kmer_index_dict[kmer]: z for kmer,z in kmer_handler.kmer_map.items()}
        self.mean    = global_mean
        self.std_dev = global_std_dev
        self.all_base_dwell_time_mean   = dwell_time_mean
        self.all_base_dwell_time_std_dev = dwell_time_std_dev

        # Define the normal distribution for dwell times
        self.getDwellSteps = lambda : max(int(np.random.normal(dwell_time_mean,dwell_time_std_dev)),1)

            
    
    def genData(self,sequence:list[int]) -> torch.Tensor:
        n = len(sequence)
        kmer_i = ''
        lmer_i = ''
        expected_current_i = 0
        current_i = 0
        current_trace = []

        # for each kmer in sequence
        for kmer_index in sequence:
        #for i,kmer_index in enumerate(sequence):
            # if the full kmer is in the sequence
            #if(n-i >= self.k):
            #    # get the kmer of relevance by slicing the sequence
            #    kmer_i             = sequence[i:i+self.k]
            #    expected_current_i = self.kmer_index_expected_zcurrent_dict[kmer_i]
            ## else (meaning that the sequence is cut off, leaving only n letters)
            #else:
            #    break
            #    # take the max over all kmers that start with the n accessible letters, and return that current
            #    lmer_i      = self.sequence[i:]
            #    expected_current_i   = self.compromiseMer(self.kmer_expected_current_dict,self.k,lmer_i)

            expected_current_i = self.kmer_index_expected_zcurrent_dict[kmer_index]
            for dwell_step in range(self.getDwellSteps()):
                current_i = np.random.normal(expected_current_i,0.2*self.std_dev)
                current_trace += [current_i]

        return torch.tensor(current_trace)
                

    def compromiseMer(self, kmer_map, k, available_sequence):
        '''returns the mean expected current of all possible kmers that start with available_sequence'''
        list_of_kmers = self.kMerTree(available_sequence,k)
        max_z = np.max(np.array([kmer_map[list_of_kmers[i]] for i in range(len(list_of_kmers))]))
        return max_z

    def kMerTree(self, available_sequence, k):
        bases = "ACGT"
        get_branch_count = lambda level, k, available_sequence : 4**(k-(level+len(available_sequence)))
        branch_counts = [get_branch_count(level, k, available_sequence) for level in range(k-len(available_sequence)+1)]
        kmer_array = branch_counts[0]*[available_sequence]
        # Get all possible kmers by looping through the indices with ACTG using modulus
        for i in range(branch_counts[0]):
            addendum = "".join([bases[(i//branch_counts[level+1])%4] for level in range(len(branch_counts)-1)])
            #print()
            kmer_array[i] = "".join([kmer_array[i][:], addendum])


        return kmer_array

    # Z scores are given in the kmer tables. This function goes from Z score to actual value
    def zToValue(self, z, current_mean, current_std_dev):
        return (z*current_std_dev + current_mean)
#OLD TESTINGk=9
#OLD TESTINGmyMap = KmerHandler('9mer_levels_v1.txt',k=k)
#OLD TESTINGmyMap.printMap(0,10)
#
#OLD TESTINGdwell_time = 5
#OLD TESTINGdataGen = FauxDataGenerator('c_elegans_chr3.txt',k=k,kmer_z_dict=myMap.getMap(),global_mean=0,global_std_dev=1)
#OLD TESTINGdata = dataGen.genData(dwell_time)
#OLD TESTINGnp.savetxt("9mer_dat.txt",data)
#OLD TESTINGsample_rate = 400 # Base pairs per second
#OLD TESTINGstep_rate = sample_rate * dwell_time
#
#OLD TESTINGplt.plot(data)
#OLD TESTINGplt.title("Simulated ONT Data")
#OLD TESTINGplt.xlabel(f"Timestep ({step_rate} steps per second)")
#OLD TESTINGplt.ylabel("pA")
#OLD TESTINGplt.show()

class Splicer():
    def __init__(self, kmer_handler: KmerHandler, sequence_filepath :str):
        self.getIntervalLength = lambda : max(int(np.random.normal(30,6)),5)
        self.kmer_handler  = kmer_handler
        self.kmer_seq, self.kmer_index_seq = self.getKmerSequenceFromFile(sequence_filepath)
        _,self.base_idx_seq = self.kmer_handler.kmerIndexSeqToBaseSeq(self.kmer_index_seq)
        self.length = len(self.kmer_index_seq)
        self.kmer_subsequences, self.base_subsequences = self.partitionSequence()



    def getSequenceFromFile(self,sequence_filepath):
        '''
        Cleans up and returns the string from the file, removing spaces and newline characters
        '''
        string = ''
        with open(sequence_filepath,'r') as f:
            while True:
                line = f.readline()
                if (line.find('#') == 0):
                    line = f.readline()
                if not line:
                    break
                string = ''.join([string,line])
            string = string.replace(' ', '')
            string = string.replace('\n', '')

        return string

    def getKmerSequenceFromFile(self,sequence_filepath) -> tuple[list[str],list[int]]:
        '''
        Retrieves a sequence of kmers (in index form) from the given file.
        '''
        k = self.kmer_handler.k
        string = self.getSequenceFromFile(sequence_filepath)
        kmer_sequence = [string[i:i+k] for i in range(len(string)-k)]
        kmer_index_sequence = [self.kmer_handler.kmer_index_dict[kmer] for kmer in kmer_sequence]
        return kmer_sequence, kmer_index_sequence

    
    def partitionSequence(self):
        kmer_subsequences = []
        base_subsequences = []
        i = 0
        while i < self.length:
            subseq_length_i = self.getIntervalLength()
            end_of_interval = i + subseq_length_i
            # Add this subsequence, or if it would extend past the length of the actual sequence, truncate it
            kmer_subsequences += [self.kmer_index_seq[i:min(end_of_interval,self.length-1)]]
            base_subsequences += [self.base_idx_seq  [i:min(end_of_interval,self.length-1)]]
            # Set the next i (non-overlapping subsequences)
            i = end_of_interval+1
        return kmer_subsequences,base_subsequences


class FauxDataset(Dataset):
    def __init__(self, kmer_map_filepath, k, load=False, sequence_filepath="none", dataset_filepath="none"):
        self.k = k
        self.kmer_handler = KmerHandler(kmer_map_filepath,k)
        self.data_generator = FauxDataGenerator(k,self.kmer_handler,global_mean=0,global_std_dev=1,dwell_time_mean=7,dwell_time_std_dev=2)
        if(load):
            self.dataset = self.loadDataset(dataset_filepath)
        else:
            self.splicer = Splicer(self.kmer_handler,sequence_filepath)
            self.dataset = self.generateDataset()



    def generateDataset(self) -> list[tuple[torch.Tensor,torch.Tensor]]:
        '''
        Returns a dataset where each element is a duple:
        tuple[0] = ground truth sequence of bases
        tuple[1] = data produced by kmer sequence
        '''
        dataset = []
        for i,subseq in enumerate(self.splicer.kmer_subsequences):
            subseq_data = self.data_generator.genData(subseq)
            dataset += [(torch.tensor(self.splicer.base_subsequences[i]),subseq_data)]
        return dataset

    def saveDataset(self,filename):
        torch.save(self.dataset,filename)

    def loadDataset(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def padded_collate(batch):
    sequences           = rnn_utils.pad_sequence([item[0] for item in batch],batch_first=True,padding_value=9)
    sequence_lengths    = torch.tensor([len(item[0]) for item in batch])

    current_traces      = rnn_utils.pad_sequence([item[1] for item in batch],batch_first=True,padding_value=9)
    trace_lengths       = torch.tensor([len(item[1]) for item in batch])


    #print("nested batch created")
    return {'sequences':sequences,'traces':current_traces,'seq_lengths':sequence_lengths,'trace_lengths':trace_lengths}


#my_dataset = FauxDataset(kmer_map_filepath=constants.kmer_map_path,sequence_filepath=constants.sequence_path,k=constants.k)
#print(my_dataset.__getitem__(10))
#my_dataset.saveDataset("worm_train_dataset")
#
#
#print("saved successfully")
#my_2nd_dataset = FauxDataset(kmer_map_filepath=constants.kmer_map_path,k=constants.k,load=True,dataset_filepath=constants.train_dataset_path + "my_dataset")
#
#my_dataloader = DataLoader(my_2nd_dataset,batch_size=3, collate_fn=padded_collate)
#
#print(my_2nd_dataset.__getitem__(0))
#
#
#panel_count = 3#my_dataset.__len__()
#
#fig = plt.figure()
#ax = fig.subplots(panel_count,1,sharex=True)
##for i in range(panel_count):
#i=0
##for i,batch in enumerate(my_dataloader):
#for batch in my_dataloader:
#    sequences = batch['sequences']
#    traces    = batch['traces']
#    for i,seq in enumerate(sequences):
#        ax[i].plot(traces[i])
#        ax[i].set_title(my_dataset.kmer_handler.kmerIndexSeqToBaseSeq(seq.tolist()))
#        ax[i].set_ylim((-2,2))
#fig.tight_layout()
#
#    
#plt.show()
###print(my_dataset[0])