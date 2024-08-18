import cv2
import dlib
import os
import face_recognition
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import mediapipe as mp
import pickle
import time
import pandas as pd
from datetime import datetime
import pywhatkit
import psutil


# Rutas de las carpetas
categories = {
    "Alumno Matriculado": "Ruta de la carpeta de los alumnos matriculados",
    "Alumno no Matriculado": "Ruta de la carpeta de los alumnos no matriculados",
    "Profesor": "Ruta de la carpeta de los profesores",
    "Trabajador": "Ruta de la carpeta de los trabajadores"
}
# Al inicio del programa
capturas_exitosas = 0
total_capturas_externas = 0

# Inicializar el contador de cuadros procesados
num_frames_procesados = 0
start_time = time.time()

def obtener_recursos():
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    ram_used_gb = ram_info.used / (1024 ** 3)  # Convertir bytes a gigabytes
    return cpu_percent, ram_percent, ram_used_gb

# Crear la ventana de Tkinter
window = tk.Tk()

# Obtener el tamaño de la pantalla
ancho_pantalla = window.winfo_screenwidth()
alto_pantalla = window.winfo_screenheight()

# Cargar la imagen de fondo y escalarla al tamaño de la pantalla
ruta_imagen = r"C:\Users\USER\Desktop\Modelo\Interfaz\AVANCE - PROTOTIPO (1).jpg"
imagen_fondo = Image.open(ruta_imagen)
imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla), Image.ANTIALIAS if hasattr(Image, 'ANTIALIAS') else Image.LANCZOS)
imagen_fondo = ImageTk.PhotoImage(imagen_fondo)

# Establecer las dimensiones de la ventana al tamaño de la pantalla
window.geometry(f"{ancho_pantalla}x{alto_pantalla}")

# Configurar el tamaño del cuadro de video y colocarlo en el centro
ancho_cuadro_video = 320  # Ajusta según el ancho deseado
alto_cuadro_video = 360  # Ajusta según el alto deseado
video_label = tk.Label(window, width=ancho_cuadro_video, height=alto_cuadro_video)
video_label.place(relx=1, rely=1, anchor=tk.CENTER)

# Crear el objeto VideoCapture
cap = cv2.VideoCapture(1)

# Crear un widget Label para la imagen de fondo
fondo_label = tk.Label(window, image=imagen_fondo)
fondo_label.place(x=0, y=0, relwidth=1, relheight=1)

# Crear el widget de la etiqueta para mostrar el video
label = tk.Label(window)
label.place(relx=0.515, rely=0.43, anchor=tk.CENTER)

dni_var = tk.StringVar()


# Configurar la entrada para el DNI
dni_entry = tk.Entry(window, textvariable=dni_var, font=("Arial", 14))
dni_entry.place(relx=0.91, rely=0.4650, anchor=tk.E)

Motivo_var = tk.StringVar()


# Configurar la entrada para el motivo de entrada
Motivo_entry = tk.Entry(window, textvariable=Motivo_var, font=("Arial", 14))
Motivo_entry.place(relx=0.91, rely=0.6450, anchor=tk.E)

def limpiar_entradas():
    dni_var.set("")  # Establecer el valor del DNI en una cadena vacía inmutable
    Motivo_var.set("")  # Establecer el valor del motivo en una cadena vacía inmutable

# Cargar las codificaciones faciales y los nombres de las imágenes
with open('La ruta de tu archivo de codificaciones facial .pickle deben ser colocadas: category_encodings.pickle', 'rb') as f:
    category_encodings = pickle.load(f)

with open('La ruta de tu archivo de etiquetas de las codificaciones facial .pickle deben ser colocadas: category_names.pickle', 'rb') as f:
    category_names = pickle.load(f)

# Inicializar mediapipe para estimar las manos, como estamos tratando de una identificacion por usuario, solo se permitira una mano
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Carpeta a donde se iran las fotos tomadas del usuario en frente en camara
ruta_destino = r'Ruta de la carpeta donde se guardaran las fotos tomadas'

def procesar_video():
    ret, frame = cap.read()
    global total_capturas_externas 
    global capturas_exitosas
    # Encontrar todas las ubicaciones de rostros en el cuadro actual
    face_locations = face_recognition.face_locations(frame, model="hog")  # Se escogio hog por ser mas rapido
    # Capturar el tiempo de inicio
    start_time_frame = time.time()

    # Verificar si se detectaron rostros
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Codificar el rostro actual
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location], model="cnn")[0]  # Se escoge CNN por ser mas preciso

            text_top = ""
            text_bottom = ""
            color = (0, 0, 255)  # Rojo para las personas cuyas codificaciones y labels no estan en los archivos .pickel
            accuracy = 0.0

            for category, encodings in category_encodings.items():
                results = face_recognition.compare_faces(encodings, face_frame_encodings, tolerance=0.5)
                if True in results:
                    # Obtener el nombre del alumno de la carpeta correspondiente
                    alumno_name = category_names[category][results.index(True)]
                    accuracy = face_recognition.face_distance(encodings, face_frame_encodings)[results.index(True)]
                    text_top = f"{category} - {alumno_name}"
                    text_bottom = f"Precisión: {1  - accuracy:.2%}"  ##No es precision en si, sino seria el acercamiento mas proximo a al codificacion facial
                    if category == "Alumno Matriculado":
                        color = (0, 255, 0)  # Verde para alumno matriculado
                    elif category == "Alumno no Matriculado":
                        color = (0, 255, 255)  # Amarillo claro para alumno no matriculado
                    elif category == "Profesor":
                        color = (255, 0, 0)  # Azul para profesor
                    elif category == "Trabajador":
                        color = (255, 255, 0)  # Celeste para trabajador
                    break 
            else:
                # Si no se encontró coincidencia en ninguna categoría, considerarlo externo con precisión del 100%
                text_top = "Externo"
                text_bottom = "Precisión: 100%"
                color = (0, 0, 255)  # Rojo

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text_top, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, text_bottom, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
    # Pasar el marco a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el marco
    result = hands.process(rgb)

    # Dibujar las manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Verificar si la mano está cerrada cando se cierra N veces, se ejecutara un script para tomar foto, esta se enviara al grupo de WSPP.
    if result.multi_hand_landmarks:
        # Obtener las coordenadas de los puntos de interés
        landmarks = result.multi_hand_landmarks[0].landmark
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
        index_finger_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
        GROUP_ID ="Coloca tu ID de grupo de wspp"       
        # Si el pulgar está a la izquierda del nudillo del dedo índice, la mano está cerrada
        if thumb_tip < index_finger_pip and len(dni_var.get()) == 8:
             total_capturas_externas += 1
             ruta_destino = r'Ruta donde ira las fotos de los usuarios externos'
       # Obtener la fecha y hora actual
             ahora = datetime.now() # Obtener la fecha y hora actual para colocarlo como metadato a la imagen
             formato_fecha_hora = ahora.strftime("%d-%m-%y_%I-%M-%p")
       # Construir el nombre del archivo con la fecha y hora
             nombre_archivo = f"{dni_var.get()}_{formato_fecha_hora}.jpg"
             nombre_ruta = os.path.join(ruta_destino, nombre_archivo)
             capturas_exitosas += 1

             cv2.imwrite(nombre_ruta, frame)
             print(f"Foto tomada y guardada como {nombre_archivo}")

             descripcion = f"DNI: {dni_var.get()}, HORA DE INGRESO: {formato_fecha_hora}, MOTIVO: {Motivo_var.get()}"
             pywhatkit.sendwhats_image(GROUP_ID, nombre_ruta, descripcion, wait_time=20, tab_close=True)
              # Limpiar las entradas
             limpiar_entradas()
    # Convertir el cuadro a formato PIL y luego a formato ImageTk
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)

    # Mostrar el cuadro en la etiqueta
    label.config(image=photo)
    label.image = photo

    window.update_idletasks()
    
      # Calcular el tiempo promedio por cuadro
    end_time_frame = time.time()
    tiempo_procesamiento_frame = end_time_frame - start_time_frame
    tiempo_procesamiento_promedio = tiempo_procesamiento_frame / num_frames_procesados if num_frames_procesados > 0 else 0

    # Obtener el uso de la CPU y la memoria RAM
    cpu_percent, ram_percent, ram_used_gb = obtener_recursos()

    # Imprimir la información en la consola
    print(f"FPS: {1 / tiempo_procesamiento_frame:.2f}")
    print(f"Tiempo de procesamiento promedio por cuadro: {tiempo_procesamiento_promedio:.4f} segundos")
    print(f"Uso de CPU: {cpu_percent:.2f}%")
    print(f"Uso de RAM: {ram_percent:.2f}% ({ram_used_gb:.2f} GB)")

    window.after(50, procesar_video)

# Iniciar el bucle principal de Tkinter
window.after(50, procesar_video)

window.mainloop()