import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import ai3


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5,
                      stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(model, loader, criterion):
    lr = 0.001
    optimizer = optim.Adam(
        model.parameters(), lr=lr)

    model.train()

    for epoch in range(5):
        total_loss_model = 0.0
        for (inputs, labels) in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss_model += loss.item()

        print(
            f'Epoch {epoch+1} - Loss: {total_loss_model / len(loader):.4f}')


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(
                outputs.data, 1)
            total += labels.size(0)
            correct += (predicted ==
                        labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    print(f'Average Loss: {avg_loss}')
    print(f'Accuracy: {accuracy:.2f}%')


dataset = datasets.MNIST(root='./datasets', train=True,
                         download=True, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=int(
    len(dataset)/100), shuffle=True)

model = ConvNet()
ai3.swap_conv2d(model)
criterion = nn.CrossEntropyLoss()

train(model, loader, criterion)
evaluate(model, loader, criterion)
