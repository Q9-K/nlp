import os
import collections
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    content = ""
    root = 'data'
    max_len = 0
    lengths = []
    
    for file in os.listdir(root):
        with open(os.path.join(root, file), 'r') as f:
            lines = f.readlines()
            content += ''.join(lines)
            lengths.extend(list(map(len, lines)))

    
    counter = collections.Counter(list(content))
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print("Total unique words:", len(sorted_counter))
    print("Top 10 most common words:")
    for word, count in sorted_counter[:10]:
        print(f"{word}: {count}")
    print("Bottom 10 least common words:")
    for word, count in sorted_counter[-10:]:
        print(f"{word}: {count}")
    
    print(np.mean(lengths), np.std(lengths), np.max(lengths), np.min(lengths))
    plt.hist(lengths, bins=50)
    plt.xlabel('Line Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Line Lengths')
    plt.savefig('line_length_distribution.png')