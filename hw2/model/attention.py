import torch
import torch.nn as nn
from model.utils.token_embedding import TokenEmbedding
from model.utils.pos_encoding import PositionalEncoding


class Attention(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1, pad_idx=0):
        super(Attention, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx
        

    def create_pad_mask(self, x):
        pad_mask = (x == self.pad_idx)
        return pad_mask
    def create_causal_mask(self, size, device):
        causal_mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        causal_mask = causal_mask.bool()
        return causal_mask
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        device = x.device
        pad_mask = self.create_pad_mask(x)
        causal_mask = self.create_causal_mask(x.size(1), device)
        x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        attn_output, _ = self.attention(x, x, x, key_padding_mask=pad_mask, attn_mask=causal_mask)  # Self-attention
        x = self.norm(x + attn_output)  # Residual connection and normalization
        x = self.ffn(x) + x  # Feed-forward network
        output = self.fc(x)  # [batch_size, seq_len, vocab_size]    
        return output