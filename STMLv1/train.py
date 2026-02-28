import time
import copy
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs=25, patience=7):
    """
    Versión mejorada con early stopping y guardado del mejor modelo
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Historial de métricas
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'val_recall': [], 'val_precision': []
    }
    
    print(f"\n{'='*60}")
    print(f"INICIANDO ENTRENAMIENTO - {epochs} épocas máximas")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        print(f"\nÉpoca {epoch+1}/{epochs}")
        print('-' * 40)
        
        # Cada época tiene una fase de entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            # Iterar sobre datos
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Optimizer zero grad
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Predicciones
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    
                    # Backward solo en entrenamiento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Estadísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Métricas de la época
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Calcular F1, recall, precision
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            if phase == 'val':
                epoch_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
                epoch_precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
                history['val_recall'].append(epoch_recall)
                history['val_precision'].append(epoch_precision)
            
            # Guardar en historial
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_f1'].append(epoch_f1)
            
            print(f"{phase.capitalize():5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}")
            
            if phase == 'val':
                print(f"           Recall (Potable): {epoch_recall:.4f} | Precision: {epoch_precision:.4f}")
        
        # Validación para early stopping (usamos F1)
        if history['val_f1'][-1] > best_f1:
            best_f1 = history['val_f1'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            print(f"  → ¡Nuevo mejor modelo! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  → Paciencia: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping en época {epoch+1}")
                break
    
    time_elapsed = time.time() - since
    print(f"\nEntrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Mejor F1: {best_f1:.4f} en época {best_epoch+1}")
    
    # Cargar mejor modelo
    model.load_state_dict(best_model_wts)
    
    return model, history


def find_best_threshold(model, dataloader, device):
    """
    Encuentra el mejor umbral para maximizar F1-score
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Probar diferentes umbrales
    thresholds = np.arange(0.2, 0.8, 0.02)
    best_f1 = 0
    best_threshold = 0.5
    results = []
    
    for thresh in thresholds:
        preds = (all_probs > thresh).astype(int)
        f1 = f1_score(all_labels, preds, average='weighted')
        recall = recall_score(all_labels, preds, pos_label=1, zero_division=0)
        precision = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'f1': f1,
            'recall': recall,
            'precision': precision
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, results