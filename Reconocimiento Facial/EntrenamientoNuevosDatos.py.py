import face_recognition
import os
import pickle

# Rutas de las carpetas
categories = {
    "Alumno Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno Matriculado",
    "Alumno no Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno no Matriculado",
    "Profesor": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Profesor",
    "Trabajador": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Trabajador"
}

# Cargar las codificaciones faciales y los nombres de las imágenes de cada categoría si ya existen
try:
    with open('category_encodings.pickle', 'rb') as f:
        category_encodings = pickle.load(f)
except FileNotFoundError:
    category_encodings = {}

try:
    with open('category_names.pickle', 'rb') as f:
        category_names = pickle.load(f)
except FileNotFoundError:
    category_names = {}

# Identificar carpetas nuevas y agregar información
for category, folder_path in categories.items():
    if category not in category_encodings:
        category_encodings[category] = []
        category_names[category] = []

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir() and os.path.basename(f.path) not in category_names[category]]
    total_subfolders = len(subfolders)
    processed_subfolders = 0

    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)
        processed_images = 0

        for img in image_files:
            face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(subfolder, img)))

            # Verificar si hay al menos una codificación facial
            if face_encodings:
                category_encodings[category].append(face_encodings[0])
                category_names[category].append(os.path.basename(subfolder))  # Añadir el nombre de la subcarpeta al diccionario

            processed_images += 1
            print(f'Procesando imagen {processed_images} de {total_images} en la subcarpeta {os.path.basename(subfolder)}')

        processed_subfolders += 1
        print(f'Procesando subcarpeta {processed_subfolders} de {total_subfolders} en la categoría {category}')

    print(f'Carpeta {category} actualizada')  # Mostrar mensaje cuando se ha terminado una categoría

# Guardar las codificaciones faciales y los nombres de las imágenes
with open('category_encodings.pickle', 'wb') as f:
    pickle.dump(category_encodings, f)

with open('category_names.pickle', 'wb') as f:
    pickle.dump(category_names, f)
