"""
Convolutional Neural Network (CNN)
Сверточная Сеть
"""

import torch
import torch.nn.functional as F     # imports activation functions
import torch.nn as nn
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from pytorch_learn_model_func import train, test


device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self, out_size, use_bn=True):
        super(CNN, self).__init__()
        # Describe layers:
        self.conv1 = nn.Conv2d(1, 32, 3, 1)     # 1: in channels, 32: out channels, 3: kernel size, 1: stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)         # 9216: in channels, 128: out channels
        self.fc2 = nn.Linear(128, out_size)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

    # Forward propagation:
    def forward(self, x):
        '''
        x reprenets our input data
        '''
        # Pass data through conv's layers:
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)

        # Run max pooling over x:
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)

        # Pass data through FC's layers:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

cnn_model = CNN(10).to(device)
print(cnn_model)


# Скачиваем данные для обучения:
training_data = datasets.MNIST(
    root="../dataset",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# Скачиваем данные для теста:
test_data = datasets.MNIST(
    root="../dataset",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

batch_size = 64

# Инициализируем модель и data loaders:
cnn_model = CNN(10).to(device)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Задаем гипперпараметры:
epochs = 5
lr = 1e-3

# Выбираем функцию потерь и оптимайзер:
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr, betas=(0.9, 0.999))

# Обучаем 5 эпох:
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, cnn_model, loss_fn, optimizer)
    test(test_dataloader, cnn_model, loss_fn)
print("Done!")


# Протестируем обученную модель:
cnn_model = cnn_model.to(device)
loss_fn = nn.CrossEntropyLoss()
test(test_dataloader, cnn_model, loss_fn)

# Опишем название классов:
classes = ['0','1','2','3','4','5','6','7','8','9']

# Делаем предсказание:
sample_idx = 444
x, y = test_data[sample_idx][0], test_data[sample_idx][1]
x = x.to(device)

def show(img, title=''):
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.show()

cnn_model.eval()
with torch.no_grad():
    pred = cnn_model(x.unsqueeze(0))
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    text = f"Prediction: {predicted} / Ground Truth: {actual}"
    show(x.cpu().squeeze(), title=text)