import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

# Inicializar Mediapipe para detección de rostro
mp_face_mesh = mp.solutions.face_mesh

# Puntos específicos del contorno del rostro en Face Mesh
face_contour_indices = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Función para clasificar el tono de piel usando K-Means y la Mediana
def classify_skin_tone_advanced(lab_image, mask):
    """Clasifica el tono de piel en categorías usando K-Means Clustering y la mediana del color."""
    skin_pixels = lab_image[mask > 0].reshape(-1, 3)

    if len(skin_pixels) == 0:
        return "unknown"

    # Aumentar el número de clusters para capturar más variabilidad
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(skin_pixels)
    cluster_centers = kmeans.cluster_centers_

    # Calcular la mediana del color de los clusters
    median_color = np.median(cluster_centers, axis=0)

    # Definir los umbrales basados en el canal L (luminosidad) para la clasificación
    L_value = median_color[0]

    if L_value > 170:
        return "Muy claro"
    elif 140 < L_value <= 170:
        return "Claro"
    elif 100 < L_value <= 140:
        return "Medio"
    elif 60 < L_value <= 100:
        return "Oscuro"
    else:
        return "Muy oscuro"

def segment_face(image, face_landmarks):
    """Segmenta la región del rostro usando los puntos de referencia de Mediapipe."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Extraer los puntos de referencia del contorno del rostro
    face_contour = [(int(face_landmarks.landmark[idx].x * image.shape[1]), 
                     int(face_landmarks.landmark[idx].y * image.shape[0])) for idx in face_contour_indices]

    # Crear una máscara de la región del rostro
    cv2.fillPoly(mask, [np.array(face_contour, dtype=np.int32)], 255)
    return mask, face_contour

def apply_illumination_correction(image):
    """Aplica la corrección de iluminación para normalizar la imagen."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    corrected_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(corrected_lab, cv2.COLOR_Lab2BGR)


# Inicializar la captura de video (cámara web o archivo de video)
cap = cv2.VideoCapture("D:\YoloV82024\Grabación 2024-08-15 190801.mp4")  # Para la cámara web, usa 0. Para un archivo de video, reemplaza con la ruta.

# Usar Face Mesh para segmentar la piel del rostro
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=20, min_detection_confidence=0.5) as face_mesh:  # Ajuste max_num_faces a un número mayor
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar corrección de iluminación
        frame = apply_illumination_correction(frame)

        # Convertir la imagen a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplicar un filtro para suavizar la imagen y reducir el ruido
        rgb_frame = cv2.bilateralFilter(rgb_frame, 9, 75, 75)

        # Procesar la imagen con Mediapipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Segmentar el rostro usando los puntos de referencia del contorno
                face_mask, face_contour = segment_face(frame, face_landmarks)

                # Aplicar la máscara para obtener solo la región de la piel
                face_roi = cv2.bitwise_and(frame, frame, mask=face_mask)

                # Convertir la imagen de la región de interés a LAB
                lab_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2Lab)

                # Clasificar el tono de piel usando K-Means y la mediana del color
                skin_tone = classify_skin_tone_advanced(lab_face, face_mask)

                # Calcular el cuadro delimitador del rostro
                x1 = min([point[0] for point in face_contour])
                y1 = min([point[1] for point in face_contour])
                x2 = max([point[0] for point in face_contour])
                y2 = max([point[1] for point in face_contour])

                # Dibujar el contorno del rostro en la imagen original
                cv2.polylines(frame, [np.array(face_contour, dtype=np.int32)], 
                             isClosed=True, color=(0, 255, 0), thickness=2)

                # Colocar el nombre del tono de piel en la imagen
                cv2.putText(frame, f'{skin_tone}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar el frame procesado
        cv2.imshow('Video en Tiempo Real', frame)

        # Presionar 'q' para salir del loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()