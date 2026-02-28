import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SuperTMLModel(nn.Module):
    """
    Modelo optimizado para SuperTML con imágenes pequeñas (64x64)
    """
    def __init__(self, num_classes=1, dropout_rate=0.4, use_pretrained=True):
        super().__init__()
        
        # Backbone ResNet18
        self.backbone = models.resnet18(pretrained=use_pretrained)
        
        # Adaptar para imágenes pequeñas (64x64 en lugar de 224x224)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Remover maxpool inicial
        
        # Obtener número de features
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Congelar capas estratégicamente
        # Solo entrenar capas superiores (layer3 y layer4)
        for name, param in self.backbone.named_parameters():
            if 'layer4' in name or 'layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Cabezal de clasificación avanzado con múltiples capas
        self.classifier = nn.Sequential(
            # Bloque 1
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Bloque 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            # Bloque 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            
            # Capa de salida
            nn.Linear(128, num_classes)
        )
        
        # Inicialización de pesos para el cabezal
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class FocalLoss(nn.Module):
    """
    Focal Loss para enfocarse en muestras difíciles
    Útil para clases desbalanceadas
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probabilidad de clasificación correcta
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


def get_model(model_type='improved', num_classes=1, **kwargs):
    """
    Fábrica de modelos
    """
    if model_type == 'improved':
        return SuperTMLModel(num_classes=num_classes, **kwargs)
    elif model_type == 'simple':
        # Modelo simple para compatibilidad
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        return model
    else:
        raise ValueError(f"Model type {model_type} not recognized")

# Mantener compatibilidad con código existente
model_res = get_model(model_type='simple')