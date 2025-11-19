import torch
import torch.nn as nn
from utils.dataset import MyDataset
from model.lstm import LSTM

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
    model = LSTM(input_size=100, hidden_size=128, num_layers=2, output_size=10, dropout=0.5)
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
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for batch in test_loader:
            inputs = batch.to(device)
            labels = torch.zeros(inputs.size(0), dtype=torch.long).to(device)  # Dummy labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    