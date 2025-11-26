import math
import torch
import torch.nn as nn
from model.utils.token_embedding import TokenEmbedding
from model.utils.pos_encoding import PositionalEncoding
import torch.nn.functional as F


# input: [batch_size, seq_len]

class LeftPadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LeftPadConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        padding = (self.kernel_size - 1, 0)  # left padding
        x_padded = F.pad(x, padding)
        out = self.conv(x_padded)
        return out

class FFN(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_dim, max_seq_len, n_gram, dropout=0.1):
        super(FFN, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.conv = LeftPadConv1d(d_model, d_model, n_gram)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        # print(x.shape)
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len]
        x = self.conv(x)  # [batch_size, d_model, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, d_model]
        # print(x.shape)
        out = self.ffn(x)  # [batch_size, seq_len, vocab_size]
        # print(out.shape)
        return out