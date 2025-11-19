from utils.tokenize_data import tokenize_text
import os

if __name__ == "__main__":
    root = 'data/'
    output = 'output/'
    os.makedirs(output, exist_ok=True)
    files = os.listdir(root)
    for file in files:
        with open(os.path.join(root, file), 'r') as f:
            lines = f.readlines()
        print(lines[0])  # Print first line for verification
        tokenized_lines = [tokenize_text(line) for line in lines]
        with open(os.path.join(output, f'tokenized_{file}'), 'w') as f:
            for tokenized_line in tokenized_lines:
                f.write(tokenized_line + '\n')