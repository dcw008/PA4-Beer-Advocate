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
        
        # TODO: CLARIFICATION, from the paper, "Our networks use 2 hidden layers with 1024 LSTM cells per layer." does it mean 
        # the hidden_size equals to 1024
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.batch_size = config['batch_size']
        self.num_layers = config['layers']
        self.lstm = nn.LSTM(input_size = cfg['input_dim'], hidden_size = config['hidden_dim'], num_layers = cfg['layers'], batch_first = True)
        
        # The linear layer that maps from hidden state space to character one hot encoding space
        self.hidden2charoh = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()
        print("init from baselineLSTM")
        
    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        # N * m * d
        """
        input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence.
        The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() 
        or torch.nn.utils.rnn.pack_sequence() for details.
        """
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        # c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        
        # output of shape (batch, seq_len, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM
        
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        # apply softmax for multi-class classification
        output = self.hidden2charoh(lstm_out.view(self.output_dim, -1))
        
        print("forward out shape", output.shape) 
        return nn.Softmax(output)
                
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # if we use one direction, num_direction equals 1
        # The axes semantics are (num_layers * num_direction, batch_size, hidden_dim)
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        