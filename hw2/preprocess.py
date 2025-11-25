from utils.tokenize_data import encode_text
import os
import numpy as np

if __name__ == "__main__":
    root = 'data/'
    output = 'output/'
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output, 'test'), exist_ok=True)
    files = os.listdir(root)
    for file in files:
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # print(lines[:100])
            train_text = lines[:-1000]
            test_text = lines[-1000:]
        train_encode = encode_text(''.join(train_text))
        test_encode = encode_text(''.join(test_text))
        np.save(os.path.join(output, 'train', file + '_train.npy'), np.array(train_encode))
        np.save(os.path.join(output, 'test', file + '_test.npy'), np.array(test_encode))
        print(f"Processed {file}, train shape: {np.array(train_encode).shape}, test shape: {np.array(test_encode).shape}")
