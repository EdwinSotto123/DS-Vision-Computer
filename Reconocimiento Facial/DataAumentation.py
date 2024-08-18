from PIL import Image, ImageEnhance
import os
import random

# Rutas de las carpetas
categories = {
    "Alumno Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno Matriculado",
    "Alumno no Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno no Matriculado",
    "Profesor": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Profesor",
    "Trabajador": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Trabajador"
}

for category, folder_path in categories.items():
    # Recorrer todas las subcarpetas en la carpeta principal
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        if os.path.isdir(subdir_path):
            # Recorrer todos los archivos en la subcarpeta
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Abrir la imagen
                    img = Image.open(os.path.join(subdir_path, filename))
                    
                    # Crear transformaciones de la imagen
                    for i in range(10):  # Crear 10 imágenes nuevas por cada imagen original
                        # Rotar la imagen un número aleatorio de grados entre -25 y 25
                        img_rotated = img.rotate(random.uniform(-25, 25))
                        
                        # Aumentar el brillo de la imagen en un factor aleatorio entre 1.0 (sin cambio) y 1.5
                        enhancer = ImageEnhance.Brightness(img_rotated)
                        img_brightened = enhancer.enhance(random.uniform(1.0, 1.5))
                        
                        # Guardar la imagen transformada
                        new_filename = f"{os.path.splitext(filename)[0]}_augmented_{i}{os.path.splitext(filename)[1]}"
                        img_brightened.save(os.path.join(subdir_path, new_filename))

print("Las imágenes transformadas han sido guardadas.")
