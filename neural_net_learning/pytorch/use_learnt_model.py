import torch
import torch.nn as nn
from pathlib import Path

from device import device
from pytorch import NeuralNetwork, test_dataloader, test_data, show
from pytorch_learn_model_func import test

BASE_DIR = Path(__file__).resolve().parent.parent
# Создаем модель (объект класса NeuralNetwork) и загружаем параметры из checkpoint:
checkpoint_path = BASE_DIR / "checkpoints/mnist_checkpoint.pth"

model = NeuralNetwork()
model.load_state_dict(torch.load(checkpoint_path))
print(model.state_dict())

model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
test(test_dataloader, model, loss_fn)


# Опишем название классов:
classes = ['0','1','2','3','4','5','6','7','8','9']

sample_idx = torch.randint(len(test_data), size=(1,)).item()
x, y = test_data[sample_idx][0], test_data[sample_idx][1]
x = x.to(device)

model.eval()
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    text = f"Prediction: {predicted} / Ground Truth: {actual}"
    show(x.cpu().squeeze(), title=text)