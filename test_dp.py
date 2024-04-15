import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import DataParallel

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_dataloader(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR100(root='./data_cifar100', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64*4, shuffle=train) # batch size与DDP保持一致
    return loader

def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    
    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx+1}, Loss: {loss.item()}")
    

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet(100).to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Utilize DataParallel
        print(f'model linear weight: {model.module.fc3.weight.data[0][0]}')
    else:
        print(f'model linear weight: {model.fc3.weight.data[0][0]}')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)

    start_time = time.time()
    for epoch in range(10):  # Number of epochs
        train(model, train_loader, optimizer, criterion, epoch, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}, Test Accuracy: {accuracy}%")
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds")

if __name__ == "__main__":
    init_seeds(0)
    main()
