import random
import torch
import numpy as np

def sequence_radom_iter(data, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    # Subtract 1 to account for the target
    num_subsequences = (len(data) - 1) // num_steps
    initial_indices = list(range(0, num_subsequences * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data_iter():
        for i in range(0, len(initial_indices), batch_size):
            batch_indices = initial_indices[i: i + batch_size]
            X = [data[j: j + num_steps] for j in batch_indices]
            Y = [data[j + 1: j + num_steps + 1] for j in batch_indices]
            yield torch.tensor(X), torch.tensor(Y)

    return data_iter()

if __name__ == "__main__":
    file = '/data/home/scyb300/run/nlp/hw2/output/test/news.2008.zh.shuffled.deduped_test.npy'
    data = np.load(file)
    data = np.arange(30)
    batch_size, num_steps = 2, 5
    data_iter = sequence_radom_iter(data, batch_size, num_steps)
    i = 0
    for X, Y in data_iter:
        print('X:', X, '\nY:', Y)
        i += 1
    print(data.shape)
    print('Total batches:', i)