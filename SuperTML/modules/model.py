import torch.nn as nn
from torchvision import models

# Cargar el modelo pre-entrenado ResNet-18
model_res = models.resnet18(pretrained=True)

# Obtener el número de características en la capa de clasificación del modelo pre-entrenado
num_features = model_res.fc.in_features

# Reemplazar la capa de clasificación (fully connected) con una nueva capa lineal para adaptarla a un problema de clasificación con 3 clases
model_res.fc = nn.Linear(num_features, 3)