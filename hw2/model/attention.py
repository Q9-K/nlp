import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)  # (batch_size, input_dim)
        return weighted_sum, attn_weights