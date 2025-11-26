import torch
import torch.nn as nn
import os
from utils.tokenize_data import encode_text, tokenizer


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_len, is_train=True):
        self.root = root
        self.max_len = max_len
        self.is_train = is_train
        files = os.listdir(root)
        train_data = []
        test_data = []
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                train_data.extend(lines[:-1000])
                test_data.extend(lines[-1000:])
        if is_train:
            self.data = train_data
        else:
            self.data = test_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = encode_text(self.data[idx])
        data = sample[:-1]
        label = sample[1:]
        if len(data) < self.max_len:
            padding_length = self.max_len - len(data)
            data += [tokenizer.pad_token_id] * padding_length
            label += [-100] * padding_length
        else:
            data = data[:self.max_len]
            label = label[:self.max_len]
        return torch.tensor(data), torch.tensor(label)
if __name__ == "__main__":
    train_dataset = MyDataset(root='data/', max_len=200, is_train=True)
    print(f'Training samples: {len(train_dataset)}')
    test_dataset = MyDataset(root='data/', max_len=200, is_train=False)
    print(f'Testing samples: {len(test_dataset)}')