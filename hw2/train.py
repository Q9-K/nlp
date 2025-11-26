import torch
import torch.nn as nn
from utils.dataset import MyDataset
from model.lstm import LSTM
from model.attention import Attention
from model.ffn import FFN
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import wandb

os.environ["WANDB_MODE"]="offline"

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
    run = wandb.init(project="nlp_hw2", name=config["model"], config=config, dir=f'output/{config["model"]}')
    set_seed(42)
    device = config['device']
    print(f'Using device: {device}')
    train_dataset = MyDataset(root='data/', max_len=config['max_seq_len'], is_train=True)
    print(f'Training samples: {len(train_dataset)}')
    sampler = RandomSampler(train_dataset, replacement=False, num_samples=config['batch_size'] * 1000)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, pin_memory=True)
    if config['model'] == 'FFN':  
        model = FFN(vocab_size=config['vocab_size'], d_model=config['embedding_dim'], hidden_dim=config['hidden_dim'], max_seq_len=config['max_seq_len'], n_gram=5, dropout=config['dropout'])
    elif config['model'] == 'LSTM':
        model = LSTM(vocab_size=config['vocab_size'], embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'], dropout=config['dropout'])
    elif config['model'] == 'Attention':
        model = Attention(vocab_size=config['vocab_size'], d_model=config['embedding_dim'], max_seq_len=config['max_seq_len'], dropout=config['dropout'])
    else:
        raise ValueError("Unknown model type")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    num_epochs = config['num_epochs']
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Batches"):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.view(-1, config['vocab_size']), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        run.log({"Train/Loss": avg_loss, "epoch": epoch})
        torch.save(model.state_dict(), f'output/{config["model"]}/model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), f'output/{config["model"]}/model_final.pth')
    run.finish()