from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False, fs=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.fs = fs
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths):
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)
        output_lstm, final_states = self.rnn(packed_sequence)
        unpadded_sequences, unpadded_lengths = pad_packed_sequence(output_lstm, batch_first=True)
        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][-2],final_states[0][-1]),dim=-1))
        else:
            h = self.dropout(final_states[0][-1].squeeze())
        if self.fs:
            y_1 = self.linear_1(h)
            return y_1
        else:
            y_1 = self.linear_1(unpadded_sequences)
            return y_1
    
