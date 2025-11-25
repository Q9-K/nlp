import torch
import torch.nn as nn
from utils.dataset import MyDataset
from model.lstm import LSTM
from model.attention import Attention
from model.ffn import FFN

config = {
    'model': 'LSTM',
    'input_size': 100,
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 10,
    'dropout': 0.5,
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    train_dataset = MyDataset(root='data/', is_train=True)
    test_dataset = MyDataset(root='data/', is_train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    model_name = config.get('model')
    if model_name == 'LSTM':
        model = LSTM(input_size=config['input_size'],
                     hidden_size=config['hidden_size'],
                     num_layers=config['num_layers'],
                     output_size=config['output_size'],
                     dropout=config['dropout'])
    elif model_name == 'Attention':
        model = Attention(input_size=config['input_size'],
                          hidden_size=config['hidden_size'],
                          output_size=config['output_size'],
                          dropout=config['dropout'])
    elif model_name == 'FFN':
        model = FFN(input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    output_size=config['output_size'],
                    dropout=config['dropout'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch.to(device)
            labels = torch.zeros(inputs.size(0), dtype=torch.long).to(device)  # Dummy labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    