
import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sns.set()
torch.manual_seed(0)
INPUT_SIZE    = 3 * 32 * 32
NUM_CLASSES   = 10
BATCH_SIZE    = 100
SAMPLE_DATA   = False
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

if SAMPLE_DATA:
    trainset, _ = torch.utils.data.random_split(trainset, [BATCH_SIZE * 10, len(trainset) - BATCH_SIZE * 10])
    valset, _ = torch.utils.data.random_split(valset, [BATCH_SIZE * 10, len(valset) - BATCH_SIZE * 10])


train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def train(net, train_loader, val_loader,
          num_epochs, learning_rate,
          compute_accs=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_accs = []
    val_accs = []
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        batch_num = 1
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_num % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_num}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

            batch_num += 1

        if compute_accs:
            train_acc = accuracy(net, train_loader)
            val_acc = accuracy(net, val_loader)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            best_val_acc = max(best_val_acc, val_acc)
            print(f'Epoch [{epoch}/{num_epochs}], Train Accuracy {100 * train_acc:.2f}%, Validation Accuracy {100 * val_acc:.2f}%')
            print(f'Best Validation Accuracy {100 * best_val_acc:.2f}%')

    if compute_accs:
        return train_accs, val_accs, best_val_acc


def accuracy(net, data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


def plot_history(histories):
    plt.figure(figsize=(16,10))
    epochs = list(range(1, len(histories[0]['train_accs']) + 1))
    for model_history in histories:
        val = plt.plot(epochs, model_history['val_accs'],
                       '--', label=model_history['name'] + ' Validation')
        plt.plot(epochs, model_history['train_accs'], color=val[0].get_color(),
                 label=model_history['name'] + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim([1, max(epochs)])
