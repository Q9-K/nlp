import torch
import torch.nn as nn
from model.utils.token_embedding import TokenEmbedding

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers = 1, dropout = 0.1):
        super(LSTM, self).__init__()
        
        self.embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        # output = self.fc(final_hidden)
        token_output = self.fc2(lstm_out)
        return token_output