# Proyecto de Detección de Actividades Sospechosas y Alertas con YOLOv8

Este proyecto tiene como objetivo la detección de personas y la identificación de actividades sospechosas como robos, utilizando un modelo preentrenado de YOLOv8 y un modelo de clasificación personalizado basado en transfer learning. Cuando se detecta una actividad sospechosa, el sistema genera una alerta y marca la región del video donde ocurre el evento.

## Archivos del Proyecto

### 1. `yolov8n.pt`
Este es el modelo preentrenado de YOLOv8 que se utiliza para detectar la clase "persona" en los videos. La detección de personas es crucial en este proyecto, ya que la información relevante se concentra alrededor de las personas dentro de un rango específico, que varía según la altura, ángulo de la cámara y la naturaleza de la escena.

### 2. `label_bin`
Este archivo contiene las etiquetas binarizadas del modelo preentrenado que se usó como base para entrenar el modelo personalizado en este proyecto. Fue obtenido y modificado a partir del repositorio: [Alert-Generation-on-Detection-of-Suspicious-Activity-using-Transfer-Learning](https://github.com/OmRajpurkar/Alert-Generation-on-Detection-of-Suspicious-Activity-using-Transfer-Learning).

### 3. `model1.h5`
Este es el modelo entrenado específicamente para este proyecto, utilizando el modelo preentrenado del repositorio mencionado anteriormente. El modelo se enfoca en clasificar diferentes actividades como "Robo", "Robo con violencia", y "Entrada forzada" basándose en las detecciones de personas y la información del entorno.

### 4. `main.py`
Este es el archivo principal que ejecuta la detección y clasificación en tiempo real utilizando los modelos cargados.

- **Flujo de Trabajo**:
  - **Cargar Modelos**: Se cargan tanto el modelo YOLOv8 como el modelo personalizado `model1.h5` para la detección y clasificación.
  - **Detección de Personas**: Utilizando YOLOv8, se detectan personas en cada cuadro del video. Los cuadros delimitadores alrededor de las personas se ajustan para capturar información contextual adicional.
  - **Clasificación de Actividades**: La región de interés (ROI) de cada persona detectada se redimensiona y procesa con el modelo `model1.h5` para clasificar la actividad que se está realizando.
  - **Generación de Alertas**: Si se detecta una actividad sospechosa, el sistema genera una alerta y marca la región en el video, indicando la actividad detectada con un cuadro de color y texto descriptivo.

- **Detalles Técnicos**:
  - **Optimización**: El uso de YOLOv8 se enfoca en la clase "persona" para reducir el ruido de detecciones irrelevantes.
  - **Preprocesamiento**: La imagen capturada en la región de interés se ajusta para cumplir con las condiciones de entrenamiento del modelo de clasificación.
  - **Tolerancia y Umbrales**: Se utilizan umbrales específicos para definir cuándo una actividad es considerada sospechosa, ajustando las etiquetas en función de condiciones temporales como el tiempo transcurrido desde el inicio del video.

## Requisitos
Para instalar todas las dependencias necesarias, asegúrate de tener un entorno de Python configurado y luego ejecuta:

```bash
pip install -r requirements.txt
```
## Demostracion
![image](https://github.com/user-attachments/assets/9917700a-d590-4d51-aead-5db299fb5315)
![image](https://github.com/user-attachments/assets/1c816dd7-ca6d-4fa7-8de7-7e92675a544d)
![image](https://github.com/user-attachments/assets/e8623537-3231-4166-8413-3634d9e73c99)


