import torch
import torch.nn as nn
import torch.optim as optim
from device import device
from pytorch import train_dataloader, test_dataloader, model

# Функция для обучения модели:
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Делаем предсказания:
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Функция для тестирования обученной модели:
def test(dataloader, model, loss_fn, verbose=True, iterations=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if iterations is not None and i >= iterations:
                break

    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# lr = 1e-3
#
# # Выбираем функцию потерь и оптимайзер:
# loss_fn = nn.CrossEntropyLoss()
# # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
#
# epochs = 5
# # Обучаем 5 эпох (эпоха - один проход по всем данным):
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")
#
# # Сохраняем checkpoint:
# checkpoint_path = '../checkpoints/mnist_checkpoint.pth'
# torch.save(model.state_dict(), checkpoint_path)
# print("Saved PyTorch Model State to {}".format(checkpoint_path))