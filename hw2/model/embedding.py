import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.embedding(input_ids)
    
if __name__ == "__main__":
    # Example usage
    vocab_size = 10000
    embedding_dim = 300
    padding_idx = 0

    model = EmbeddingLayer(vocab_size, embedding_dim, padding_idx)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    embeddings = model(input_ids)
    print(embeddings.shape)  # Should print: torch.Size([2, 3, 300])