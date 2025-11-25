import torch
import torch.nn as nn
from model.utils.token_embedding import TokenEmbedding

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        
        self.embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = hidden[-1]
        output = self.fc(final_hidden)
        token_output = self.fc2(final_hidden)
        return output, token_output