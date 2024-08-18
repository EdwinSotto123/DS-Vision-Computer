import cv2
import numpy as np
from ultralytics import YOLO
import os   
# Cargar los modelos de YOLOv8
model_segmentation = YOLO("deepfashion2_yolov8s-seg.pt")
model_person = YOLO(r"yolov8n.pt")

# Definir las clases
class_names = [
    "camisa_manga_corta", "camisa_manga_larga", "abrigo_manga_corta", 
    "abrigo_manga_larga", "Chaleco", "Top_con_tirantes", "shorts", "pantalones", 
    "falda", "vestido_manga_corta", "vestido_manga_larga", "vestido_tipo_chaleco", "vestido_con_tirantes"
]

# Definir rangos de color en HSV (más extensos y específicos)
color_ranges = {
    "red": [(0, 50, 50), (10, 255, 255)],     # Rojo
    "red2": [(170, 50, 50), (180, 255, 255)], # Segundo rango de rojo
    "green": [(36, 50, 50), (85, 255, 255)],  # Verde
    "light_green": [(85, 50, 50), (100, 255, 255)],  # Verde claro
    "blue": [(90, 50, 50), (130, 255, 255)],  # Azul
    "light_blue": [(85, 50, 50), (100, 255, 255)],  # Celeste
    "yellow": [(25, 50, 50), (35, 255, 255)], # Amarillo
    "cyan": [(85, 50, 50), (100, 255, 255)],  # Cian
    "magenta": [(140, 50, 50), (160, 255, 255)], # Magenta
    "pink": [(160, 50, 50), (170, 255, 255)], # Rosado
    "white": [(0, 0, 200), (180, 30, 255)],   # Blanco
    "black": [(0, 0, 0), (180, 255, 50)],     # Negro
    "gray": [(0, 0, 50), (180, 50, 200)],     # Gris
    "orange": [(10, 50, 50), (25, 255, 255)], # Naranja
    "purple": [(130, 50, 50), (140, 255, 255)], # Púrpura
    "beige": [(20, 20, 80), (30, 100, 255)],   # Beige
    "brown": [(10, 100, 20), (20, 255, 200)],  # Marrón
    "mint": [(150, 20, 50), (165, 255, 255)], # Verde Menta
    "olive": [(50, 50, 50), (70, 255, 100)],  # Verde Oliva
    "teal": [(85, 50, 50), (100, 255, 255)],  # Azul Verdoso
    "turquoise": [(160, 50, 50), (180, 255, 255)], # Turquesa
    "violet": [(130, 50, 50), (145, 255, 255)], # Violeta
}

def classify_color_hsv(hsv_color):
    """Clasifica un color en HSV basado en rangos predefinidos."""
    h, s, v = hsv_color

    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        if lower_bound[0] <= h <= upper_bound[0] and lower_bound[1] <= s <= upper_bound[1] and lower_bound[2] <= v <= upper_bound[2]:
            return color_name.replace('2', '')  # Para manejar el caso de 'red' y 'red2'
    
    return "unknown"

def detect_dominant_color_histogram(image, mask):
    """Detecta el color dominante usando histogramas de color en un objeto segmentado."""
    # Aplicar la máscara para obtener los píxeles relevantes
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convertir la imagen segmentada a HSV
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # Inicializar un diccionario para contar los píxeles por color
    color_counts = {color_name: 0 for color_name in color_ranges}

    # Clasificar cada píxel según su color
    for y in range(hsv.shape[0]):
        for x in range(hsv.shape[1]):
            if mask[y, x] > 0:  # Si el píxel está dentro de la máscara
                color_name = classify_color_hsv(hsv[y, x])
                if color_name != "unknown":
                    color_counts[color_name] += 1

    # Encontrar el color con mayor número de píxeles
    dominant_color_name = max(color_counts, key=color_counts.get)
    dominant_color_hsv = np.mean([np.mean(color_ranges[dominant_color_name], axis=0)], axis=0)
    
    # Convertir HSV a BGR para visualización
    dominant_color_bgr = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

    return tuple(map(int, dominant_color_bgr)), dominant_color_name
def get_mask_center(mask):
    """Calcula el centroide de la prenda segmentada usando la máscara."""
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return center_x, center_y
    else:
        return None, None
    
def remove_overlap(mask1, mask2):
    """Elimina la superposición entre dos máscaras."""
    return cv2.subtract(mask1, mask2)

# Cargar la imagen en la que se va a realizar la inferencia
image_path = r"ruta de imagen" # Reemplaza con la ruta a tu imagen
frame = cv2.imread(image_path)

# Detección de personas en la imagen
person_results = model_person(frame, classes=[0], verbose=False)

def process_image(image_path, output_folder):
    """Procesa una imagen para detectar colores dominantes y guarda el resultado."""
    frame = cv2.imread(image_path)
    
    # Detección de personas en la imagen
    person_results = model_person(frame, classes=[0], verbose=False)
    
    for person_result in person_results:
        boxes = person_result.boxes.xyxy  # Coordenadas de las cajas (xmin, ymin, xmax, ymax)
        scores = person_result.boxes.conf  # Confianza de las detecciones

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Recortar la región del frame correspondiente a la persona detectada
            person_roi = frame[y1:y2, x1:x2]

            # Realizar la segmentación en la región recortada
            segmentation_results = model_segmentation(person_roi, verbose=False)

            # Guardar todas las máscaras para procesar la eliminación de superposición
            masks = []
            cls_list = []

            for seg_result in segmentation_results:
                if seg_result.masks is not None and seg_result.boxes.cls is not None:
                    for mask, cls in zip(seg_result.masks.data, seg_result.boxes.cls):
                        mask = mask.squeeze().cpu().numpy().astype(np.uint8)
                        mask = cv2.resize(mask, (person_roi.shape[1], person_roi.shape[0]))
                        masks.append(mask)
                        cls_list.append(cls)

            # Eliminar superposiciones en el orden inverso de las detecciones (para asegurar que las prendas superiores no interfieran con las inferiores)
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    masks[i] = remove_overlap(masks[i], masks[j])

            # Procesar cada máscara ajustada
            for mask, cls in zip(masks, cls_list):
                # Detectar el color dominante de la prenda usando histogramas de color
                dominant_color_bgr, color_name = detect_dominant_color_histogram(person_roi, mask)

                # Crear una imagen en blanco del mismo tamaño que la ROI para colorear la máscara
                colored_mask = np.zeros_like(person_roi, dtype=np.uint8)
                colored_mask[mask > 0] = dominant_color_bgr

                # Reemplazar la prenda original con la prenda coloreada en la ROI
                person_roi[mask > 0] = colored_mask[mask > 0]

                # Obtener el centroide de la prenda para colocar la etiqueta de color
                center_x, center_y = get_mask_center(mask)
                if center_x is not None and center_y is not None:
                    center_x += x1  # Ajustar posición al marco original
                    center_y += y1

                    # Colocar el nombre del color en el centro de la prenda
                    cv2.putText(frame, color_name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Volver a insertar la región segmentada en el frame original
        frame[y1:y2, x1:x2] = person_roi

    # Guardar la imagen procesada
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, frame)

    print(f"Imagen procesada guardada en: {output_path}")

# Lista de imágenes a procesar
image_paths = [
    r"ruta de imagen",
    r"ruta de imagen",
    r"ruta de imagen",
    r"ruta de imagen"
]

# Carpeta de salida para las imágenes procesadas
output_folder = r"ruta al folder de salida"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Procesar cada imagen en la lista
for image_path in image_paths:
    process_image(image_path, output_folder)
