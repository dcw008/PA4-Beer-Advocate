import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class baselineLSTM(nn.Module):
    def __init__(self, config):
        super(baselineLSTM, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        self.hidden_dim = config['hidden_dim']
        self.minibatch_size = config['batch_size']
        self.num_layers = config['layers']
        self.lstm = nn.LSTM(input_size = cfg['input_dim'], hidden_size = config['hidden_dim'], num_layers = cfg['layers'])
        
        
        
        print("init from baselineLSTM")
        
    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        # N * m * d after for the processed train data
        
        
        
        print("forward")
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))
        