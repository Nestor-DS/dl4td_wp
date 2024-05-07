from PIL import Image, ImageDraw, ImageFont
import numpy as np

def show_column_position(dat):
    """
    Crea una imagen para visualizar la posición de las columnas de datos.
    
    Args:
        dat (list): Lista de datos.
    
    Returns:
        np.array: Matriz numpy que representa la imagen.
    """
    font = ImageFont.truetype("arial.ttf", size=20)
    background = np.array([[0 for _ in range(255)] for _ in range(255)], dtype='uint8')
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    draw.text((32, 32), str(dat[0][:12]), fill='white', font=font)
    draw.text((32, 160), str(dat[1][:11]), fill='white', font=font)
    draw.text((160, 32), str(dat[2][:11]), fill='white', font=font)
    draw.text((160, 160), str(dat[3][:11]), fill='white', font=font)
    rgb = [np.array(image, dtype='uint8') for _ in range(3)]
    return np.array(rgb)

def show_column_position_another(dat):
    """
    Crea otra imagen para visualizar la posición de las columnas de datos.
    
    Args:
        dat (list): Lista de datos.
    
    Returns:
        np.array: Matriz numpy que representa la imagen.
    """
    font = ImageFont.truetype("arial.ttf", size=20)
    background = np.array([[0 for _ in range(255)] for _ in range(255)], dtype='uint8')
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    draw.text((32, 32), str(dat[2][:12]), fill='white', font=font)
    draw.text((32, 160), str(dat[3][:11]), fill='white', font=font)
    draw.text((160, 32), str(dat[1][:11]), fill='white', font=font)
    draw.text((160, 160), str(dat[0][:11]), fill='white', font=font)
    rgb = [np.array(image, dtype='uint8') for _ in range(3)]
    return np.array(rgb)
    
def data_to_image(data):
    """
    Convierte datos en imágenes.
    
    Args:
        data (list): Lista de datos.
    
    Returns:
        np.array: Matriz numpy que representa las imágenes.
    """
    data_images = []
    font = ImageFont.truetype("arial.ttf", size=20)
    for dat in data:
        background = np.array([[0 for _ in range(255)] for _ in range(255)], dtype='uint8')
        image = Image.fromarray(background)
        draw = ImageDraw.Draw(image)
        draw.text((40, 190), str(dat[0]), fill='white', font=font)
        draw.text((120, 190), str(dat[1]), fill='white', font=font)
        draw.text((190, 190), str(dat[2]), fill='white', font=font)
        
        draw.text((40, 120), str(dat[3]), fill='white', font=font)
        draw.text((120, 120), str(dat[4]), fill='white', font=font)
        draw.text((190, 120), str(dat[5]), fill='white', font=font)
        
        draw.text((40, 40), str(dat[6]), fill='white', font=font)
        draw.text((110, 40), str(dat[7]), fill='white', font=font)
        draw.text((190, 40), str(dat[8]), fill='white', font=font)
        
        rgb = [np.array(image, dtype='uint8') for _ in range(3)]
        data_images.append(rgb)
    return np.array(data_images) / 255

def data_to_image_another(data):
    """
    Convierte datos en imágenes de otra manera.
    
    Args:
        data (list): Lista de datos.
    
    Returns:
        np.array: Matriz numpy que representa las imágenes.
    """
    data_images = []
    font = ImageFont.truetype("arial.ttf", size=50)
    for dat in data:
        background = np.array([[0 for _ in range(255)] for _ in range(255)], dtype='uint8')
        image = Image.fromarray(background)
        draw = ImageDraw.Draw(image)
        draw.text((32, 32), str(dat[2]), fill='white', font=font)
        draw.text((32, 160), str(dat[3]), fill='white', font=font)
        draw.text((160, 32), str(dat[1]), fill='white', font=font)
        draw.text((160, 160), str(dat[0]), fill='white', font=font)
        rgb = [np.array(image, dtype='uint8') for _ in range(3)]
        data_images.append(rgb)
    return np.array(data_images) / 255

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from PIL import Image
    data = load_iris()
    img_array = data_to_image(data.data)
    img = Image.fromarray(img_array)
    img.show()
