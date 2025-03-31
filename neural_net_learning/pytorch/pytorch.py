"""
Как при помощи PyTorch:

- Работать с данными
- Создавать свои модели нейросетей (на примере полносвязных и сверточных сетей)
- Обучать созданные модели
- Сохранять и загружать обученные модели (checkpoints)
- Делать предсказания (инференс)
- Ускорить инференс при помощи компиляции моделей (PyTorch 2.0)
- Transfer Lerning из одного датасета в другой


В качестве датасетов будем использовать MNIST и CIFAR10
MNIST:
70 000 изображений рукописных цифр от 0 до 9
— 60 000 для обучения
— 10 000 для тестирования

CIFAR10:
60 000 цветных изображений из 10 категорий
(например, самолеты, машины, кошки, собаки, птицы, корабли и тд)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

print(f"{torch.__version__}")

def show(img, title=''):
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.show()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from device import device

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

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") # squeeze() - удаляем все измерения размером 1
# plt.show()


batch_size = 64

# Создаем data loaders:
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")


# Определяем кастомную модель (Sequential):
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=(28,28,1), output_size=10):
        super(NeuralNetwork, self).__init__()  # Вызываем конструктор родительского класса nn.Module
        self.flatten = nn.Flatten()  # Преобразует входное изображение в одномерный вектор
        self.fc1 = nn.Linear(input_size[0]*input_size[1]*input_size[2], 512)  # Полносвязный слой: 784 входа → 512 нейронов
        self.fc2 = nn.Linear(512, 512)  # Полносвязный слой: 512 входов → 512 нейронов
        self.relu = nn.ReLU(inplace=True)  # Функция активации ReLU, заменяет отрицательные значения на 0
        self.output = nn.Linear(512, output_size)  # Выходной слой: 512 входов → количество классов (например, 10)

    def forward(self, x):
        x = self.flatten(x)  # Преобразует входное изображение размером [batch_size, 28, 28] в вектор [batch_size, 784]
        x = self.fc1(x)  # Прогоняет данные через первый полносвязный слой (784 → 512)
        x = self.relu(x)  # Применяет ReLU активацию к выходу первого слоя (делает отрицательные значения 0)
        x = self.fc2(x)  # Прогоняет данные через второй полносвязный слой (512 → 512)
        x = self.relu(x)  # Применяет ReLU активацию ко второму скрытому слою
        logits = self.output(x)  # Прогоняет данные через последний слой, который преобразует 512 в выходное количество классов

        return logits  # Возвращает логиты (необработанные предсказания), которые можно использовать для дальнейшей классификации

"""
[784 входов]
   ↓ Linear → 512 нейронов
   ↓ ReLU
   ↓ Linear → 512 нейронов
   ↓ ReLU
   ↓ Linear → 10 выходов (классы)
"""

model = NeuralNetwork().to(device)
print(model)

# # Мы можем итерироваться через параметры (веса) нашей модели:
# for name, param in model.named_parameters():
#     print(f"Имя: {name} | Размер: {param.size()} | Значения : {param[:2]} \n")
#

# Пропустим рандомное изображение из MNIST через нашу еще необученную модель:
sample_idx = torch.randint(len(training_data), size=(1,)).item()
X, Y = training_data[sample_idx]
X = X.to(device)

# Forward pass:
logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits)
Y_pred = pred_prob.argmax(1)

print('Ненормализованные вероятности:\n', logits.cpu().detach())
print('Нормализованные вероятности:\n', pred_prob.cpu().detach())
text = f"Prediction: {Y_pred.item()} / Ground Truth: {Y}"
# show(X.cpu().squeeze(), title=text)

