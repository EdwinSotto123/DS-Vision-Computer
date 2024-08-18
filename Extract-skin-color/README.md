# Proyecto de Clasificación de Tono de Piel

Este proyecto tiene como objetivo identificar y clasificar el tono de piel en cinco categorías distintas utilizando un video en tiempo real o un archivo de video previamente grabado. La detección y clasificación se realiza mediante el uso de Mediapipe para la segmentación del rostro y el algoritmo K-Means para la clasificación de colores en el espacio de color LAB.

## Explicación Detallada del Proyecto

### 1. **Detección de Rostros con Mediapipe**
El proyecto utiliza la solución Face Mesh de Mediapipe para detectar rostros en video en tiempo real. Mediapipe Face Mesh es una técnica avanzada que permite detectar 468 puntos de referencia faciales en 3D con alta precisión. Para este proyecto, se utilizan específicamente los puntos del contorno del rostro, lo que permite segmentar la región de la piel del rostro para un análisis más detallado.

### 2. **Segmentación del Rostro**
Una vez detectados los puntos de referencia del contorno del rostro, el script `skin_color.py` crea una máscara que cubre solo la región de la piel del rostro. Esta máscara permite aislar la piel de otras partes del rostro, como ojos, labios o cabello, lo que es crucial para la correcta clasificación del tono de piel.

### 3. **Corrección de Iluminación**
Para mejorar la precisión de la clasificación, se aplica una corrección de iluminación a la imagen capturada. La corrección se realiza en el espacio de color LAB, donde el canal L (luminosidad) se iguala utilizando histogramas. Esto normaliza las diferencias de iluminación en la imagen, garantizando que el análisis del color no se vea afectado por sombras o reflejos.

### 4. **Conversión a Espacio de Color LAB**
El espacio de color LAB es utilizado en este proyecto debido a su capacidad para representar los colores de manera que se asemejen más a la percepción humana del color. En LAB, el canal L representa la luminosidad, mientras que los canales A y B representan las dimensiones de color (verde-rojo y azul-amarillo, respectivamente). Esta representación es más robusta para analizar el color de la piel bajo diferentes condiciones de iluminación.

### 5. **Clasificación del Tono de Piel con K-Means**
Para clasificar el tono de piel, el proyecto emplea el algoritmo de agrupamiento K-Means en los valores de color LAB de los píxeles de la piel. K-Means agrupa los píxeles en varios clústeres, y se calcula la mediana del color de los centros de estos clústeres. El valor de luminosidad (L) del color mediano se utiliza para categorizar el tono de piel en una de las cinco clases: Muy Claro, Claro, Medio, Oscuro, o Muy Oscuro.

### 6. **Visualización de Resultados**
Finalmente, el tono de piel detectado se muestra superpuesto en el video en tiempo real. La región segmentada del rostro se destaca y se etiqueta con la categoría correspondiente. Esto permite una visualización clara y precisa del análisis de tono de piel.

### 7. **GIF de Demostración**
El proyecto incluye un GIF (`output.gif`) que muestra el resultado en acción. Este GIF se encuentra en el mismo directorio que `skin_color.py` y proporciona una demostración visual del proyecto en funcionamiento.

![Demostración del Proyecto](output.gif)

## Archivos del Proyecto

### 1. `skin_color.py`
Este es el script principal del proyecto y realiza todas las tareas mencionadas anteriormente, desde la captura de video y la detección de rostros hasta la clasificación y visualización del tono de piel.

### 2. `output.gif`
Este GIF de demostración muestra el resultado del proyecto en acción, donde los tonos de piel se clasifican y etiquetan en tiempo real.

## Requisitos Previos

- Python 3.7 o superior.
- Las dependencias necesarias listadas en `requirements.txt`.
### Instalación

1. Clona este repositorio en tu máquina local.
2. Instala las dependencias necesarias ejecutando:
   ```bash
   pip install -r requirements.txt
   python skin_color.py
