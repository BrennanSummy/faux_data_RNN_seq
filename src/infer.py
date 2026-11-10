import constants
import Model
import dataSimTests
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def getLengthsAferCNN(input_lengths):
    lengths = (input_lengths + 2*constants.c1_pad - constants.c1_kernel_size) / 2
    return lengths

def easyDecode(output:torch.Tensor):
    predicted_classes = torch.argmax(output,dim=1)
    pred_list = predicted_classes.tolist()
    final = []
    for i in range(len(pred_list)):
        p_im1 = pred_list[i-1]
        p_i   = pred_list[i]
        if(p_i != p_im1 and p_i != 0):
            final += [p_i]
    return final


device = torch.device("xpu")

model = Model.Seq_LSTMwCTC(5).to(device)

my_params = torch.load("weights/r4_end_epoch200_weights.pth")
model.load_state_dict(my_params)
model.eval()


#datSet = dataSimTests.FauxDataset(constants.kmer_map_path,constants.k, load=True,dataset_filepath=constants.test_dataset_path + "test_dataset")
datSet = dataSimTests.FauxDataset(constants.kmer_map_path,constants.k, load=True,dataset_filepath=constants.train_dataset_path + "worm_train_dataset")
# CLEARLY OVERFITTING

batch_size = 1
datLoad = DataLoader(datSet, batch_size, True, collate_fn=dataSimTests.padded_collate)

seq_idx, trace = datSet.__getitem__(19)
input_length = torch.tensor([len(trace)])
post_cnn_length = getLengthsAferCNN(input_length)

input = (trace.unsqueeze(0)).unsqueeze(0).to(device)

#print(input.shape)

out, output_lengths = model(input, post_cnn_length)
#print(out.shape)
out = out.squeeze(0)
#print(out.shape)
print(easyDecode(out))
#probs = torch.softmax(out, dim=1)
print(seq_idx.tolist())
#predicted_classes = torch.argmax(out, dim=1)
#print(predicted_classes)
my_kmer_handler = dataSimTests.KmerHandler(map_filepath=constants.kmer_map_path,k=constants.k)

#pred_class_list = predicted_classes.tolist()[0]
#print(pred_class_list)
#predicted_label = dataSet.decode(pred_class_list)
#print(f'Predicted label: {predicted_label}')
#print(f'True label: {true_label}')

fig = plt.figure()
ax = fig.subplots()
ax.plot(np.array(trace))
true_seq = [my_kmer_handler.idx_to_ACGT_dict[base_idx] for base_idx in seq_idx.tolist()]
predicted_seq = [my_kmer_handler.idx_to_ACGT_dict[base_idx] for base_idx in easyDecode(out)]
title = "True: " + "".join(true_seq) + "\nPred: " + "".join(predicted_seq)
ax.set_title(title)
ax.set_ylim((-2,2))
fig.tight_layout()

    
plt.show()
##print(my_dataset[0])