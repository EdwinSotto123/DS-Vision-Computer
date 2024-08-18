import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(layout="wide")
st.title("Video Capture and Knive Detection with YOLO")

# Placeholder para el video o la imagen
frame_placeholder = st.empty()

# Opción para elegir el modelo
model_choice = st.selectbox("Select YOLO Model", ("Heavy Model", "Light Model"))

# Definir rutas de los modelos usando os.path.join
current_dir = os.path.dirname(os.path.abspath(__file__))
heavy_model_path = os.path.join(current_dir, 'modelos', 'YoloHeavyV8.pt')
light_model_path = os.path.join(current_dir, 'modelos', 'YoloLightV8.pt')

# Cargar el modelo YOLO basado en la elección del usuario
try:
    if model_choice == "Heavy Model":
        yolo_model = YOLO(heavy_model_path)
    else:
        yolo_model = YOLO(light_model_path)
    st.success(f"{model_choice} loaded successfully!")
except Exception as e:
    st.error(f"Error loading modeal: {e}")

# Opción para elegir la fuente de medios
media_source = st.selectbox("Select Media Source", ("Webcam", "Upload Video", "Upload Image", "Select from Predefined"))

def process_frame(frame, yolo_model):
    results = yolo_model(frame)
    for result in results:
        # Obtener las clases, confianzas y detecciones de la caja
        classes = result.names
        # obtener las clases, confianzas y detecciones de la caja
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy
        # Dibujar las predicciones en el frame
        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                color = (0, int(cls[pos]) * 10 % 255, 255)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

#FUNCION PARA PROCESAR VIDEO
def display_video(video_path, yolo_model, width=480):
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("The video capture has ended.")
            break

        # procesar el frame usando YOLO
        frame = process_frame(frame, yolo_model)

        # cONVERTIR EL FRAME DE BGR A RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # mOSTRAR EL FRAME USANDO ST.IMAGE
        frame_placeholder.image(frame, channels="RGB", use_column_width=True, width=width)

    cap.release()
    cv2.destroyAllWindows()

# FUNCION PARA PROCESAR IMAGEN
def display_image(image_path, yolo_model, width=480):
    if not os.path.exists(image_path):
        st.error(f"Image file not found: {image_path}")
        return

    image = Image.open(image_path)
    frame = np.array(image)

   # Convertir RGB a formato BGR para OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame usando YOLO
    frame = process_frame(frame, yolo_model)

    # Convertir RGB a formato BGR para OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MOSTRAR LA IMAGEN USANDO ST.IMAGE
    frame_placeholder.image(frame, channels="RGB", use_column_width=True, width=width)

# Rutas a videos e imágenes predefinidas
predefined_videos = [
    os.path.join(current_dir, 'media', 'videos', 'RUTA DE TUS VIDEOS'),
    os.path.join(current_dir, 'media', 'videos', 'RUTA DE TUS VIDEOS')
]
predefined_images = [
    os.path.join(current_dir, 'media', 'imagenes', 'RUTA DE TUS IMAGENES'),
    os.path.join(current_dir, 'media', 'imagenes', 'RUTA DE TUS IMAGENES')
]

# Mostrar opciones predefinidas en una cuadrícula 2x2
if media_source == "Select from Predefined":
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(predefined_images[0]):
            st.image(predefined_images[0], caption="Predefined Image 1", use_column_width=True, width=300)
            if st.button("Process Predefined Image 1"):
                display_image(predefined_images[0], yolo_model, width=300)
        else:
            st.error(f"Predefined Image 1 not found: {predefined_images[0]}")

    with col2:
        if os.path.exists(predefined_images[1]):
            st.image(predefined_images[1], caption="Predefined Image 2", use_column_width=True, width=300)
            if st.button("Process Predefined Image 2"):
                display_image(predefined_images[1], yolo_model, width=300)
        else:
            st.error(f"Predefined Image 2 not found: {predefined_images[1]}")

    with col1:
        if os.path.exists(predefined_videos[0]):
            st.video(predefined_videos[0], format="video/mp4", start_time=0)
            if st.button("Process Predefined Video 1"):
                display_video(predefined_videos[0], yolo_model, width=300)
        else:
            st.error(f"Predefined Video 1 not found: {predefined_videos[0]}")

    with col2:
        if os.path.exists(predefined_videos[1]):
            st.video(predefined_videos[1], format="video/mp4", start_time=0)
            if st.button("Process Predefined Video 2"):
                display_video(predefined_videos[1], yolo_model, width=300)
        else:
            st.error(f"Predefined Video 2 not found: {predefined_videos[1]}")

# Si el usuario selecciona Webcam
elif media_source == "Webcam":
    start_webcam = st.button("Start Webcam")
    if start_webcam:
        cap = cv2.VideoCapture(0)
        stop_button_pressed = st.button("Stop Webcam")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended.")
                break

            # Procesar el frame usando YOLO
            frame = process_frame(frame, yolo_model)

            # Convertir el frame de BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mostar el frame usando st.image
            frame_placeholder.image(frame, channels="RGB", use_column_width=True, width=480)

            # Romper el bucle si el usuario hace clic en el botón "Stop"
            if stop_button_pressed:
                break

        cap.release()
        cv2.destroyAllWindows()

# Si el usuario selecciona Upload Video
elif media_source == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file....", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Guardar el video subido en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            temp_video_path = temp_video_file.name
        process_video_button = st.button("Process Video")
        if process_video_button:
            display_video(temp_video_path, yolo_model, width=480)

# Si el usuario selecciona Upload Image
elif media_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Guardar la imagen subida en un archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image_file:
            temp_image_file.write(uploaded_file.read())
            temp_image_path = temp_image_file.name
        process_image_button = st.button("Process Image")
        if process_image_button:
            display_image(temp_image_path, yolo_model, width=480)
