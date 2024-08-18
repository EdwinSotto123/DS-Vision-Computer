import face_recognition
import os
import pickle

# Rutas de las carpetas
categories = {
    "Alumno Matriculado": "Ruta a tus carpeta con imagenes de alumnos matriculados",
    "Alumno no Matriculado": "Ruta a tus carpeta con imagenes de alumnos no matriculados",
    "Profesor": "Ruta a tus carpeta con imagenes de profesores",
    "Trabajador": "Ruta a tus carpeta con imagenes de trabajadores"
}
# La estructura de las carpetas sera asi:
# -Carpeta raiz con las carpetas con imagenes
## -- Carpeta llamada: Alumnos matriculado
## --- Carpeta por cada alumno matriculado: Alumno matriculado 1
###---- imagen1.jpg
# Diccionario para almacenar las codificaciones faciales y los nombres de las imágenes de cada categoría
category_encodings = {}
category_names = {}

# Este bucle for esta disenhado para trabajar con N categorias.

# Cargar las codificaciones faciales y los nombres de las imágenes de cada categoría
for category, folder_path in categories.items():
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    total_subfolders = len(subfolders)
    used_subfolders = 0
    category_encodings[category] = []
    category_names[category] = []

    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img in image_files:
            face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(subfolder, img)))
            
            # Verificar si hay al menos una codificación facial
            if face_encodings:
                category_encodings[category].append(face_encodings[0])
                category_names[category].append(os.path.basename(subfolder))  # Añadir el nombre de la subcarpeta al diccionario

        used_subfolders += 1
        # Calcular el indicador de avance
        progress = used_subfolders / total_subfolders
        print(f'Avance: {progress * 100}%')
        print(f'Subcarpeta "{os.path.basename(subfolder)}" acabada')  # Mostrar mensaje cuando se ha terminado una subcarpeta

    print(f'Carpeta {category} acabada')  # Mostrar mensaje cuando se ha terminado una categoría

# Guardar las codificaciones faciales y los nombres de las imágenes
with open('category_encodings.pickle', 'wb') as f:
    pickle.dump(category_encodings, f)

with open('category_names.pickle', 'wb') as f:
    pickle.dump(category_names, f)

