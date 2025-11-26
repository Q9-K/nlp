import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import MyDataset
from model.lstm import LSTM
from model.attention import Attention
from model.ffn import FFN
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

config = {
    'model': None,
    'vocab_size': 21128,
    'embedding_dim': 512,
    'hidden_dim': 1024,
    'max_seq_len': 200,
    'dropout': 0.1,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'num_epochs': 200,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model type: LSTM, Attention, or FFN')
    args = parser.parse_args()
    config['model'] = args.model
    os.makedirs(f'output/{config["model"]}', exist_ok=True)
    set_seed(42)
    device = config['device']
    print(f'Using device: {device}')
    writer = SummaryWriter(log_dir='logs')
    test_dataset = MyDataset(root='data/', max_len=config['max_seq_len'], is_train=False)
    print(f'Testing samples: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    if config['model'] == 'FFN':  
        model = FFN(vocab_size=config['vocab_size'], d_model=config['embedding_dim'], hidden_dim=config['hidden_dim'], max_seq_len=config['max_seq_len'], n_gram=5, dropout=config['dropout'])
    elif config['model'] == 'LSTM':
        model = LSTM(vocab_size=config['vocab_size'], embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], n_layers=2, dropout=config['dropout'])
    elif config['model'] == 'Attention':
        model = Attention(vocab_size=config['vocab_size'], d_model=config['embedding_dim'], max_seq_len=config['max_seq_len'], dropout=config['dropout'])
    else:
        raise ValueError("Unknown model type")
    checkpoint_path = f'output/{config["model"]}/model_final.pth'
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    num_epochs = config['num_epochs']
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config['vocab_size']), targets.view(-1))
            total_loss += loss.item()
            writer.add_scalar('Test/Batch_Loss', loss.item())
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')
    writer.close()