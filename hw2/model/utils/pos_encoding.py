import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.embedding(position_ids)  # (batch_size, seq_len, d_model)