# Proyecto de Detección y Clasificación de Colores Dominantes en Prendas de Vestir

Este proyecto está diseñado para detectar y clasificar los colores dominantes en prendas de vestir, utilizando modelos de segmentación y detección de personas. La aplicación es especialmente útil en el ámbito de la moda y la seguridad, donde identificar prendas y sus colores puede ser crucial.

## Archivos del Proyecto

### 1. `Extract_color.py`
Este script es el núcleo del proyecto, encargado de procesar imágenes para detectar personas y segmentar las prendas de vestir que usan. Luego, clasifica y visualiza el color dominante de cada prenda detectada. Los principales pasos del script incluyen:

- **Carga de Modelos**:
  - `deepfashion2_yolov8s-seg.pt`: Modelo de segmentación entrenado en el dataset DeepFashion2, especializado en la detección y segmentación de diferentes tipos de prendas de vestir.
  - `yolov8n.pt`: Modelo YOLOv8, utilizado para detectar personas en las imágenes.
  
- **Definición de Clases y Rangos de Color**:
  - El script define un conjunto de clases relacionadas con prendas de vestir, como "camisa_manga_corta" o "falda".
  - Los rangos de colores en el espacio de color HSV se configuran para identificar colores comunes como rojo, verde, azul, amarillo, entre otros, con gran precisión.

- **Clasificación de Colores Dominantes**:
  - Utilizando histogramas de color, el script clasifica los colores dominantes de las prendas segmentadas.
  - Cada prenda es evaluada en función de su color, que es determinado a partir de los píxeles dentro de la máscara segmentada.

- **Eliminación de Superposición**:
  - Para asegurar que los colores y prendas no se superpongan incorrectamente, se implementa una función para eliminar la superposición entre diferentes máscaras de prendas detectadas.

- **Visualización y Guardado**:
  - El script coloca el nombre del color dominante directamente sobre la prenda segmentada en la imagen procesada.
  - Las imágenes procesadas se guardan en una carpeta de salida definida por el usuario.

### 2. `deepfashion2_yolov8s-seg.pt`
Este es un modelo preentrenado utilizado para la segmentación de prendas de vestir. Fue entrenado en el dataset DeepFashion2, que contiene una amplia variedad de categorías de ropa, lo que lo hace ideal para este tipo de aplicaciones.

### 3. `yolov8n.pt`
El modelo YOLOv8 se utiliza en este proyecto para la detección de personas en las imágenes. Esto es crucial, ya que la segmentación de prendas se realiza solo dentro de las regiones donde se han detectado personas.

## Cómo Ejecutar el Proyecto

### Requisitos Previos
- Python 3.7 o superior.
- Dependencias instaladas en `requirements.txt`.

### Pasos para Ejecutar

1. **Preparar el Entorno**:
   - Instala las dependencias necesarias:
     ```bash
     pip install -r requirements.txt
     ```

2. **Ejecutar el Script**:
   - Asegúrate de que las rutas a las imágenes, modelos y carpetas de salida están correctamente configuradas en el script `Extract_color.py`.
   - Ejecuta el script:
     ```bash
     python Extract_color.py
     ```

3. **Resultados**:
   - Las imágenes procesadas se guardarán en la carpeta especificada como `output_folder`, con las prendas segmentadas y los colores dominantes etiquetados.
