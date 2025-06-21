class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 300)
        self.fc2 = nn.Linear(300, NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=25, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(25 * 14 * 14, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 25 * 14 * 14)
        x = self.fc(x)
        return x

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
