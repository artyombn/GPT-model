import torch

# Выбираем девайс (cpu или gpu), на котором будут происходит вычисления:
device = "cuda" if torch.cuda.is_available() else "cpu"
# cuda - технология от NVIDIA, которая позволяет запускать вычисления на видеокарте
print(f"Using {device} device")