import torch
import torch.nn as nn
import os
from tokenize_data import tokenize_text
from sample import sequence_radom_iter

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train=True):
        self.root = root
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
            self.data = "".join(train_data)
            self.data = tokenize_text(self.data)
        else:
            self.data = "".join(test_data)
            self.data = tokenize_text(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    def __repr__(self):
        return f"MyDataset(root={self.root}, is_train={self.is_train}, length={len(self)})"
    def get_sample(self, idx):
        return self.__getitem__(idx)
    def get_all_data(self):
        return self.data

if __name__ == "__main__":
    dataset = MyDataset(root='data/', is_train=True)
    print(dataset.get_sample(0))
    print(f"Total samples: {len(dataset)}")