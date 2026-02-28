import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
import time

def normalize_features(X):
    """
    Normaliza características a rango [0,1] de forma eficiente
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    # Evitar división por cero
    range_vals = X_max - X_min
    range_vals[range_vals == 0] = 1
    X_normalized = (X - X_min) / range_vals
    return X_normalized.astype(np.float32)

def data_to_image_fast(X, img_size=64):
    """
    VERSIÓN CORREGIDA: maneja correctamente las dimensiones
    """
    n_samples, n_features = X.shape
    print(f"Transformando {n_samples} muestras a imágenes {img_size}x{img_size}...")
    start_time = time.time()
    
    # Normalizar
    X_norm = normalize_features(X)
    
    # Pre-asignar memoria
    images = np.zeros((n_samples, 3, img_size, img_size), dtype=np.float32)
    cell_size = img_size // 3
    
    # Crear índices una sola vez
    rows = np.arange(img_size).reshape(1, -1, 1)
    cols = np.arange(img_size).reshape(1, 1, -1)
    
    for sample_idx in range(n_samples):
        # Crear imagen para esta muestra (3 canales)
        img = np.zeros((3, img_size, img_size))
        
        for feature_idx in range(n_features):
            valor = X_norm[sample_idx, feature_idx]
            
            # Posición en cuadrícula 3x3
            row = feature_idx // 3
            col = feature_idx % 3
            
            # Coordenadas de la celda
            y_start = row * cell_size
            y_end = (row + 1) * cell_size
            x_start = col * cell_size
            x_end = (col + 1) * cell_size
            
            # Crear máscara para esta celda (2D)
            mask_2d = np.zeros((img_size, img_size))
            mask_2d[y_start:y_end, x_start:x_end] = 1
            
            # Calcular centro de la celda
            center_y = (y_start + y_end) / 2
            center_x = (x_start + x_end) / 2
            
            # Calcular distancias al centro (resultado 2D)
            dist_2d = np.sqrt((rows - center_y)**2 + (cols - center_x)**2).squeeze()
            max_dist = cell_size / np.sqrt(2)
            
            # Intensidad basada en distancia (2D)
            intensidad_2d = (1 - dist_2d / max_dist).clip(0, 1) * valor
            
            # ¡CORRECCIÓN! Multiplicar elemento a elemento (ambos 2D)
            contribucion = intensidad_2d * mask_2d
            
            # Aplicar a cada canal RGB con diferente énfasis
            # IMPORTANTE: contribucion ya es (64,64), igual que img[0]
            img[0] += contribucion * (0.9 if feature_idx % 3 == 0 else 0.5)
            img[1] += contribucion * (0.9 if feature_idx % 3 == 1 else 0.5)
            img[2] += contribucion * (0.9 if feature_idx % 3 == 2 else 0.5)
        
        images[sample_idx] = img
        
        # Mostrar progreso
        if (sample_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Progreso: {sample_idx + 1}/{n_samples} muestras ({elapsed:.1f}s)")
    
    total_time = time.time() - start_time
    print(f" Transformación completada en {total_time:.1f} segundos")
    return images

def data_to_image_alternativa(X, img_size=64):
    """
    VERSIÓN ALTERNATIVA MÁS SIMPLE: si la anterior sigue dando problemas
    """
    n_samples, n_features = X.shape
    print(f"Usando versión alternativa para {n_samples} muestras...")
    
    X_norm = normalize_features(X)
    images = np.zeros((n_samples, 3, img_size, img_size))
    cell_size = img_size // 3
    
    for idx in range(n_samples):
        img = np.zeros((3, img_size, img_size))
        
        for f in range(n_features):
            val = X_norm[idx, f]
            r, c = f // 3, f % 3
            
            # Coordenadas
            y1, y2 = r * cell_size, (r + 1) * cell_size
            x1, x2 = c * cell_size, (c + 1) * cell_size
            
            # Asignar valor directamente (más simple)
            img[0, y1:y2, x1:x2] = val * 0.8
            img[1, y1:y2, x1:x2] = val * 0.6
            img[2, y1:y2, x1:x2] = val * 0.4
        
        images[idx] = img
        
        if (idx + 1) % 100 == 0:
            print(f"  Progreso: {idx + 1}/{n_samples}")
    
    return images

def save_images_to_cache(images, filename):
    """Guarda imágenes en cache"""
    os.makedirs('cache', exist_ok=True)
    cache_path = os.path.join('cache', filename)
    np.save(cache_path, images)
    print(f" Imágenes guardadas en {cache_path}")

def load_images_from_cache(filename):
    """Carga imágenes desde cache si existen"""
    cache_path = os.path.join('cache', filename)
    if os.path.exists(cache_path + '.npy'):
        print(f" Cargando imágenes desde cache: {filename}")
        return np.load(cache_path + '.npy')
    return None

def prepare_water_data(X_train, X_val, force_recompute=False, use_alternative=False):
    """
    Función principal: prepara datos con cache automático
    """
    cache_file = f'water_images_{len(X_train)}_{64}.npy'
    
    if not force_recompute:
        cached = load_images_from_cache(cache_file)
        if cached is not None:
            n_train = len(X_train)
            return cached[:n_train], cached[n_train:]
    
    print("=== TRANSFORMANDO DATOS A IMÁGENES ===")
    X_all = np.vstack([X_train, X_val])
    
    # Elegir versión
    if use_alternative:
        images_all = data_to_image_alternativa(X_all, img_size=64)
    else:
        images_all = data_to_image_fast(X_all, img_size=64)
    
    # Guardar en cache
    save_images_to_cache(images_all, cache_file)
    
    # Separar
    n_train = len(X_train)
    return images_all[:n_train], images_all[n_train:]

# Mantener compatibilidad
data_to_image = data_to_image_fast