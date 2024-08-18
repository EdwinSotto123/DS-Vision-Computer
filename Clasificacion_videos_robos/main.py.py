import cv2
import numpy as np
import pickle
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import os
from deep_sort.tracker import Tracker
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Forzar el uso de la CPU

# Configuración de rutas y parámetros
model_path = r"Ruta a modelo model1.h5"
label_bin_path = r"Ruta a label_bin"
input_video_path = r"Ruta del video de entrada"
output_video_path = r"Ruta del video de salida"
queue_size = 10

# Cargar el modelo y el label binarizer
print("[INFO] Cargando modelo sin compilar...") ### Ya que .h5 tiene la configuración de la red, no es necesario compilarlo
model = load_model(model_path, compile=False)

lb = pickle.loads(open(label_bin_path, "rb").read())

# Inicializar la media de la imagen para la resta de media junto con la cola de predicciones
# SE DEBE AJUSTAR ESTA MEDIA DEPENDIENDO DE LA RED CON LA QUE SE ENTRENÓ EL MODELO
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
Q = deque(maxlen=queue_size)

# Cargar el modelo YOLOv8
yolo_model = YOLO("yolov8n.pt")

# Obtener el índice de la clase "person" en el modelo YOLOv8
person_class_id = None
for class_id, class_name in yolo_model.names.items():
    if class_name == "person":
        person_class_id = class_id
        break

# Asegurarse de que se encontró el ID de la clase "person"
if person_class_id is None:
    raise ValueError("La clase 'person' no se encuentra en el modelo YOLOv8")

# Inicializar el flujo de video, puntero al archivo de video de salida y dimensiones del cuadro
vs = cv2.VideoCapture(input_video_path)
writer = None
(W, H) = (None, None)
prelabel = '' #Creando un array vacio
ok = 'Normal' #Inicializando como Normal
fi_label = []
framecount = 0
start_time = time.time()
normal_mode_duration = 0  # Debido al ruido de los videos al inicio, puedes modificar esto para evadir X segundos al inicio
# Establecer el nombre de la ventana y su tamaño
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# Función para calcular IoU para unir cuadros delimitadores (aun no se usa debido a lo pesado del modelo)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Calcular la intersección
    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Calcular la unión
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


# Bucle sobre cuadros del flujo de archivo de video
while True:
    # Leer el siguiente cuadro del archivo
    (grabbed, frame) = vs.read()

    # Si el cuadro no fue tomado, hemos llegado al final del flujo
    if not grabbed:
        break

    # Si las dimensiones del cuadro están vacías, tomarlas
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        cv2.resizeWindow("Output", W, H)  # Ajustar la ventana al tamaño del video

    framecount += 1
    potential_robbery_detected = False # Variable para determinar si se detecta algún robo

    # Calcular tiempo transcurrido
    elapsed_time = time.time() - start_time # Calcular el tiempo transcurrido

    # Detección de personas usando YOLOv8
    results = yolo_model(frame)

    # Filtrar detecciones de personas y ampliar los cuadros delimitadores
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item()) # Obtener la clase de la detección

            # Filtrar solo personas
            if cls == person_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Aumentar el cuadro delimitador
                width = x2 - x1 # Calcular el ancho del cuadro para tener la variable al cual poder multiplicar y modificar el tamano de las cajas
                height = y2 - y1 # Calcular la altura del cuadro
                x1 = max(0, x1 - int(2 * width))  # Ampliar 15% a la izquierda
                x2 = min(frame.shape[1], x2 + int(2 * width))  # Ampliar 15% a la derecha
                y1 = max(0, y1 - int(0.20 * height))  # Ampliar 10% hacia arriba
                y2 = min(frame.shape[0], y2 + int(0.20 * height))  # Ampliar 10% hacia abajo

                # Extraer la región de interés (ROI) para usar el modelo .h5 para hacer predicciones solo sobre estas regiones y asi evitar ruido del entorno
                roi = frame[y1:y2, x1:x2]

                # Redimensionar la ROI al tamaño esperado por el modelo de Keras
                roi_resized = cv2.resize(roi, (224, 224)).astype("float32")
                roi_resized -= mean

                # Hacer predicciones con el modelo de Keras
                preds = model.predict(np.expand_dims(roi_resized, axis=0), verbose=0)[0]

                prediction = preds.argmax(axis=0)
                Q.append(preds)

                # Realizar el promedio de predicciones
                results = np.array(Q).mean(axis=0)
                maxprob = np.max(results)
                predicted_index = np.argmax(results)
                label = lb[predicted_index]
                rest = 1 - maxprob
                diff = maxprob - rest # Para determinar si se encuentra de lo normal
                th = 100
                if diff > 0.80:
                    th = diff

                # Ajustar etiquetas según las reglas
                if elapsed_time <= normal_mode_duration and label == 'Shoplifting':
                    label = 'Normal'
                elif label == 'Shoplifting':
                    label = 'Robo'
                if elapsed_time <= normal_mode_duration and label == 'Robbery':
                    label = 'Normal'
                elif label == 'Robbery':
                    label = 'Robo'
                if elapsed_time <= normal_mode_duration and label == 'Break-In':
                    label = 'Normal'
                elif label == 'Shoplifting':
                    label = 'Robo'

                # Evaluar si la predicción cumple con el umbral
                if maxprob > 0.8 and label == 'Robo':
                    potential_robbery_detected = True
                    text = "Alerta: {} - {:.2f}%".format(label, maxprob * 100)
                    color = (0, 0, 255) if label == 'Robo' else (0, 255, 0)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    text = "Alerta: Normal - {:.2f}%".format(maxprob * 100)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Dibujar el cuadro delimitador
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar el mensaje general en la parte superior de la pantalla
    if elapsed_time <= normal_mode_duration:
        cv2.putText(frame, "ZONA NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif potential_robbery_detected:
        cv2.putText(frame, "!!!POTENCIAL ROBO!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "ZONA NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Comprobar si el escritor de video es None
    if writer is None:
        # Inicializar nuestro escritor de video
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_video_path, fourcc, 30, (W, H), True)

    # Escribir el cuadro de salida en disco
    writer.write(frame)

    # Mostrar la imagen de salida
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # Si se presionó la tecla `q`, romper el bucle
    if key == ord("q"):
        break

print('Frame count', framecount)
print('Count label', fi_label)

# Liberar los punteros de archivo
print("[INFO] cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows() 