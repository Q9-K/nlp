import torch
import torch.nn as nn
from model.utils.token_embedding import TokenEmbedding
from model.utils.pos_encoding import PositionalEncoding

class Attention(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super(Attention, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model] for MultiheadAttention
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.fc(attn_output)  # [batch_size, seq_len, vocab_size]
        return output