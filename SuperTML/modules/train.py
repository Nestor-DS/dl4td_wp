import copy
import time
import torch
import torch.nn as nn

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs, scheduler = None):
    """
    Función para entrenar un modelo de red neuronal.

    Args:
        model (torch.nn.Module): Modelo de red neuronal a entrenar.
        dataloaders (dict): Diccionario que contiene los dataloaders para entrenamiento y validación.
        dataset_sizes (dict): Diccionario que contiene el tamaño de los conjuntos de datos de entrenamiento y validación.
        criterion: Función de pérdida.
        optimizer: Optimizador.
        device: Dispositivo de cómputo (CPU o GPU) para entrenamiento.
        scheduler: (Opcional) Scheduler para ajustar la tasa de aprendizaje.
        num_epochs (int): Número de épocas de entrenamiento (por defecto: 1).

    Returns:
        model (torch.nn.Module): Modelo entrenado.
    """
    start_time = time.time()
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}/{num_epochs}:")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            
            # Set model mode (train/eval)
            model.train(phase == 'train')
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler:
                scheduler.step()
            
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        
        print()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {(total_time // 60):.0f}m {(total_time % 60):.0f}s")
    print(f"BEST VALIDATION ACCURACY: {best_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model

