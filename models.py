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
        self.config = config
        self.lstm = nn.LSTM(input_size = config['input_dim'], hidden_size = config['hidden_dim'], num_layers = config['layers'], batch_first = True)
        
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

        type of the sequency <class 'torch.Tensor'>
        type of the sequency torch.Size([50, 2075, 217])
        lstm out of the sequency <class 'torch.Tensor'>
        lstm out of the sequency torch.Size([50, 2075, 100])
        """
        lstm_out, self.hidden = self.lstm(sequence.cuda(), self.hidden)
        # lstm out of the sequency torch.Size([50, 2075, 100])
        output = self.hidden2charoh(lstm_out.reshape([lstm_out.size(0)*lstm_out.size(1), -1]).cuda())
        output = output.view(sequence.shape[0], sequence.shape[1], -1)
        return output # don't need to apply activation function if we use crossEntropyLoss during training
                
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # if we use one direction, num_direction equals 1
        # The axes semantics are (num_layers * num_direction, batch_size, hidden_dim)
        if self.config['cuda']:
            print("cuda availabe")
            computing_device = torch.device("cuda")
        else:
            print("cuda not availabe")
            computing_device = torch.device("cpu")
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device = computing_device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device = computing_device))
        