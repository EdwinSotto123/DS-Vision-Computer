# Proyecto de Clasificación de Tomates: Detección de Enfermedades

Este proyecto tiene como objetivo crear un modelo de red neuronal convolucional (CNN) para clasificar imágenes de tomates en dos categorías: **sano** y **enfermo**. El proyecto utiliza diversas técnicas de procesamiento de imágenes y aprendizaje profundo para entrenar y evaluar el modelo.

## Descripción del Proyecto

### 1. **Carga y Preprocesamiento de Imágenes**
El proyecto comienza con la carga de imágenes de tomates desde un conjunto de datos, que se divide en carpetas de entrenamiento, validación y prueba. Se utiliza la librería PIL para cargar y explorar las dimensiones de las imágenes, y se emplean técnicas de preprocesamiento, como la normalización de píxeles, para preparar las imágenes para el entrenamiento.

### 2. **Aumento de Datos**
Para mejorar la capacidad del modelo para generalizar, se implementa la técnica de aumento de datos (`ImageDataGenerator`). Este proceso incluye transformaciones como:
- Rotaciones aleatorias.
- Desplazamientos horizontales y verticales.
- Zoom aleatorio.
- Inversión horizontal.

Estas transformaciones ayudan a crear variaciones adicionales de las imágenes de entrenamiento, lo que permite que el modelo sea más robusto frente a variaciones en las imágenes reales.

### 3. **Construcción del Modelo CNN**
El modelo se construye utilizando la API `Sequential` de Keras, con múltiples capas convolucionales y de agrupación (`Conv2D` y `MaxPooling2D`). Además, se utilizan técnicas como la normalización por lotes (`BatchNormalization`) y el abandono (`Dropout`) para mejorar el rendimiento del modelo y prevenir el sobreajuste. Las capas de la red incluyen:
- Capas convolucionales para la extracción de características.
- Capas de pooling para la reducción de dimensionalidad.
- Capas de activación ReLU para la no linealidad.
- Capas densas para la clasificación final con una activación sigmoide para salida binaria.

### 4. **Entrenamiento del Modelo**
El modelo se entrena utilizando un conjunto de datos de entrenamiento, y su rendimiento se valida en un conjunto de datos de validación. Se utiliza `binary_crossentropy` como función de pérdida y `RMSprop` como optimizador. Además, se implementa una reducción del aprendizaje automático (`ReduceLROnPlateau`) para ajustar dinámicamente la tasa de aprendizaje según el rendimiento del modelo.

### 5. **Evaluación del Modelo**
Después del entrenamiento, el modelo se evalúa en el conjunto de prueba. Las métricas de rendimiento incluyen la precisión y la pérdida, que se visualizan a través de gráficos que muestran la variación entre el conjunto de entrenamiento y el de validación a lo largo de las épocas.

### 6. **Visualización de Kernels y Feature Maps**
Para entender mejor lo que ha aprendido el modelo, se visualizan los kernels de la primera capa convolucional y los mapas de características (feature maps) generados por las primeras capas convolucionales para una imagen de prueba.

### 7. **Predicción en Nuevas Imágenes**
Se desarrollan funciones para cargar y preprocesar nuevas imágenes, y luego realizar predicciones utilizando el modelo entrenado. Estas funciones permiten predecir si un tomate está sano o enfermo, basándose en imágenes de entrada.

### 8. **Clasificación y Reporte de Resultados**
El proyecto genera un informe de clasificación que resume el rendimiento del modelo en términos de precisión, sensibilidad, especificidad, y otras métricas clave. Este informe es crucial para evaluar la efectividad del modelo en la tarea de clasificación.

### 9. **Guardado del Modelo**
Finalmente, el modelo entrenado se guarda en un archivo `model.h5` para ser utilizado posteriormente en aplicaciones de predicción.
