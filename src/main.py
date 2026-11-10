import constants
import dataSimTests
import Model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


def getLengthsAferCNN(input_lengths):
    lengths = (input_lengths + 2*constants.c1_pad - constants.c1_kernel_size) / 2
    return lengths

datSet = dataSimTests.FauxDataset(constants.kmer_map_path,constants.k, load=True,dataset_filepath=constants.train_dataset_path + "worm_train_dataset")

batch_size = 10
datLoad = DataLoader(datSet, batch_size, True, collate_fn=dataSimTests.padded_collate)

model = Model.Seq_LSTMwCTC(5)

my_params = torch.load("weights/r2_end_epoch300_weights.pth")
model.load_state_dict(my_params)

device = torch.device('xpu')

ctc = nn.CTCLoss(blank=0).to(device)
optimizer = optim.AdamW(model.parameters(),lr=0.002)

model = model.to(device)

num_epochs = 301


for epoch in range(num_epochs):
    model.train()

    for batch_index, batch in enumerate(datLoad):
        # load inputs and outputs onto GPU
        current_traces = batch['traces'].to(device)
        #print(batch['label_indices'])
        base_idx_seqs = batch['sequences'].to(device)
        input_lengths = batch['trace_lengths']
        post_cnn_lengths = torch.tensor(getLengthsAferCNN(input_lengths))

        # clear grad
        optimizer.zero_grad()

        # forward
        outputs, output_lengths = model(current_traces.unsqueeze(1), post_cnn_lengths)
        outputs = outputs.permute(1,0,2)
        #print(outputs[0])
        #print(base_idx_seqs[0])

        #output_lengths = torch.full(size=(batch_size,), fill_value=9, dtype= torch.int).to(device)
        label_lengths = batch['seq_lengths'].to(device)
        #print(images.shape)
        #print(outputs.shape)
        #print(label_indices.shape)
        #print(input_lengths.shape)
        #print(label_lengths)

        # get loss
        loss = ctc(outputs,base_idx_seqs,output_lengths,label_lengths)

        # make sure it's reasonable
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'loss went nan, skipping step: {loss.item()}')
            continue

        # calculate gradients
        loss.backward()

        optimizer.step()
        if(batch_index%25==0):
            print(f"Loss: {loss.item():.4f}. Batch {batch_index}/{len(datLoad)}. Epoch {epoch}")

        if((batch_index==len(datLoad)-1) and (epoch%50==0)):
            torch.save(model.state_dict(),f'weights/r4_end_epoch{epoch}_weights.pth')