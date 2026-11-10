import constants
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq_LSTMwCTC(nn.Module):
    def __init__(self, number_of_output_classes, input_dim=1, convolution_channels=128, hidden_size=256, lstm_layer_count=3):
        super(Seq_LSTMwCTC,self).__init__()

        # Feature extraction via 1D convolution
        self.convolution_layer1 = nn.Conv1d(input_dim,convolution_channels, kernel_size=constants.c1_kernel_size, padding=constants.c1_pad)
        self.batch_norm1        = nn.BatchNorm1d(convolution_channels)
        self.maxpool1           = nn.MaxPool1d(kernel_size=3,stride=constants.stride,padding=1)

        self.convolution_layer2 = nn.Conv1d(convolution_channels,2*convolution_channels, kernel_size=3, padding=1)
        self.batch_norm2        = nn.BatchNorm1d(2*convolution_channels)
        self.maxpool2           = nn.MaxPool1d(kernel_size=2,stride=2*constants.stride)

        # Normalize over convolution features to stabilize the network over training
        self.layer_norm_layer = nn.LayerNorm(2*convolution_channels)

        # Linear fully connected layer between the conv and RNN
        self.fc_cnn_to_rnn = nn.Linear(2*convolution_channels, convolution_channels*2)

        # LSTM (RNN)
        self.lstm_layer = nn.LSTM(2*convolution_channels,hidden_size=hidden_size,num_layers=lstm_layer_count,
                                  batch_first=True,bidirectional=True)

        # Linear fully connected layer
        self.fully_connected_layer = nn.Linear(hidden_size*2, number_of_output_classes)

        # Dropout layer (has a chance to zero out elements)
        self.dropout = nn.Dropout(0.1)

        #### Bidirectional Gated Recurrent Unit, which looks at the sequence of features forwards and backwards
        ###self.gru_layer = nn.GRU(convolution_channels,hidden_gru_channels,gru_layer_count,
        ###                        bidirectional=True, batch_first=False, dropout=0.2)

        # Get the bidirectional output (2 * hidden_gru_channels) and try to pull out the class from there 
        ###self.fully_connected_layer = nn.Linear(2*hidden_gru_channels,number_of_output_classes)


    def forward(self, x, post_cnn_lengths):
        # x should have shape [batch size, input_dim, steps]
        #if constants.DEBUG: print(f'start: {x.shape}')
        x = self.convolution_layer1(x)
        #if constants.DEBUG: print(f'after conv 1: {x.shape}')
        x = self.batch_norm1(x)
        #if constants.DEBUG: print(f'after batch norm 1: {x.shape}')
        x = F.relu(x)
        x = self.maxpool1(x)
        #if constants.DEBUG: print(f'after maxpool 1: {x.shape}')
        x = self.convolution_layer2(x)
        #if constants.DEBUG: print(f'after conv 2: {x.shape}')
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        #if constants.DEBUG: print(f'after maxpool 2: {x.shape}')

        x = x.permute(0,2,1)
        #if constants.DEBUG: print(f'after permute: {x.shape}')
        x = self.layer_norm_layer(x)
        x = self.fc_cnn_to_rnn(x)
        #if constants.DEBUG: print(f'after linear conversion layer: {x.shape}')

        x = nn.utils.rnn.pack_padded_sequence(x,post_cnn_lengths,batch_first=True,enforce_sorted=False)
        # After convolutions we have [batch size, convolution_channels, timestep_feature_count]
        # Linear layer wants conv channels in last idx
        # Rearrange into few columns, each of which is considered a timestep
        #batch_size, features, channels = x.shape
        #x = x.permute(0,2,1,3)
        #x = x.reshape(batch_size,width,features*height)
        #print(x.shape)
        #print(x.shape)

        x,_ = self.lstm_layer(x)
        x,out_lengths = nn.utils.rnn.pad_packed_sequence(x,batch_first=True)
        #if constants.DEBUG: print(f'after lstm: {x.shape}')

        x = self.fully_connected_layer(x)
        x = self.dropout(x)
        #print(x.shape)


        return F.log_softmax(x,dim=-1),out_lengths
