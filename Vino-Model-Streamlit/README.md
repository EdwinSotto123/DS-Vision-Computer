# Proyecto de Detección de Armas con YOLOv8 y Streamlit

Este proyecto está diseñado para detectar armas de fuego y armas blancas en imágenes y videos en tiempo real utilizando el modelo YOLOv8 optimizado con OpenVINO. La aplicación se despliega usando Streamlit, lo que facilita la interacción del usuario y la visualización de los resultados.

## Archivos del Proyecto

### 1. `Dockerfile`
El `Dockerfile` contiene las instrucciones necesarias para construir una imagen de Docker que ejecuta la aplicación. Incluye la instalación de las dependencias necesarias para OpenCV y la configuración de Streamlit para correr la aplicación.

- **Componentes Clave**:
  - **Instalación de Dependencias**: Configura el entorno de Python y las librerías del sistema necesarias para ejecutar OpenCV.
  - **Copiado de Archivos**: Copia los archivos del proyecto y los modelos necesarios dentro del contenedor Docker.
  - **Configuración de Exposición de Puertos**: Expone el puerto `8080` donde se ejecutará la aplicación Streamlit.
  - **Comando de Ejecución**: Inicia la aplicación utilizando Streamlit en el contenedor Docker.

### 2. `Optimizar-Yolo`
Este script optimiza el modelo YOLOv8 utilizando OpenVINO para reducir su peso y mejorar el rendimiento en la inferencia.

- **Funciones Clave**:
  - **Exportación a OpenVINO**: Convierte el modelo YOLOv8 preentrenado a un formato optimizado para ejecutarse con OpenVINO.
  - **Reducción del Peso del Modelo**: Ayuda a mejorar el tiempo de procesamiento, haciéndolo más eficiente para la detección en tiempo real.

### 3. `app.yaml`
El archivo `app.yaml` es utilizado para configurar el despliegue de la aplicación en Google Cloud Platform (GCP) utilizando Cloud Run.

- **Configuración Clave**:
  - **Entorno Flex**: Especifica el entorno flexible de Google App Engine, permitiendo personalizar el entorno de ejecución.
  - **Runtime Custom**: Define el uso de un entorno personalizado basado en el Dockerfile.

### 4. `app.py`
Este es el archivo principal de la aplicación Streamlit que maneja la lógica de la detección y la interfaz de usuario.

- **Flujo de Trabajo**:
  - **Selección de Modelos**: Permite al usuario seleccionar entre diferentes versiones del modelo YOLOv8 (Heavy y Light).
  - **Fuente de Medios**: El usuario puede seleccionar entre varias fuentes de medios, como la webcam, subir videos o imágenes, o usar contenido predefinido.
  - **Procesamiento en Tiempo Real**: Usa el modelo seleccionado para detectar armas en tiempo real en la fuente de medios elegida.
  - **Visualización de Resultados**: Muestra las predicciones directamente en el navegador utilizando la interfaz de Streamlit.
