# Proyecto de Detección de Rostros y Autenticación Mediante Gestos

Este proyecto está diseñado para la detección de rostros de personas utilizando una base de datos previamente codificada y etiquetada. El sistema permite la autenticación mediante un gesto específico (cerrar el puño tres veces frente a la cámara), y captura fotos de personas no reconocidas, añadiendo metadatos y enviándolas a un grupo de WhatsApp.

## Archivos del Proyecto

### 1. `DataAumentation.py`
Este script se utiliza para aumentar la cantidad de datos de entrenamiento a partir de las imágenes originales. Realiza transformaciones aleatorias en las imágenes, como rotaciones y ajustes de brillo, generando múltiples versiones de cada imagen para mejorar la robustez del modelo de reconocimiento facial.

- **Características Clave**:
  - Rota las imágenes en un rango aleatorio de -25 a 25 grados.
  - Ajusta el brillo en un rango de 1.0 a 1.5 veces el brillo original.
  - Genera 10 nuevas imágenes por cada imagen original, guardándolas en la misma carpeta de origen.

### 2. `Entrenamiento.py`
Este archivo se encarga de entrenar el modelo de reconocimiento facial utilizando las imágenes codificadas y organizadas en categorías. Las imágenes se procesan para obtener sus codificaciones faciales, las cuales se almacenan junto con los nombres de las personas en archivos `.pickle`.

- **Estructura de Carpetas**:
  - Cada categoría (por ejemplo, "Alumno Matriculado", "Profesor") tiene una carpeta con subcarpetas individuales para cada persona.
  - Las imágenes dentro de estas subcarpetas se usan para crear codificaciones faciales que luego se compararán durante la detección.

- **Proceso de Entrenamiento**:
  - Carga las imágenes de cada categoría.
  - Genera codificaciones faciales utilizando la librería `face_recognition`.
  - Guarda las codificaciones y los nombres en archivos `.pickle` para su posterior uso.

### 3. `main.py`
Este es el archivo principal que ejecuta el sistema de detección y autenticación en tiempo real. Utiliza la cámara conectada al sistema para capturar el video, detectar rostros y realizar la autenticación mediante gestos.

- **Funcionalidad Principal**:
  - **Detección de Rostros**: Utiliza la librería `face_recognition` para detectar rostros en el video en tiempo real y compararlos con las codificaciones almacenadas.
  - **Autenticación por Gestos**: Implementa Mediapipe para detectar la mano levantada y reconocer cuando el usuario cierra el puño tres veces.
  - **Captura y Envío de Fotos**: Si el rostro no es reconocido, el sistema toma una foto y la guarda junto con metadatos como motivo y fecha. Esta foto se envía automáticamente a un grupo de WhatsApp.

- **Interfaz Gráfica**:
  - Desarrollada usando `Tkinter` para mostrar el video en tiempo real, capturar entradas como DNI y motivo de ingreso, y gestionar la interfaz del usuario.

- **Uso de Recursos del Sistema**:
  - Monitorea el uso de CPU y RAM, imprimiendo esta información en la consola para un control eficiente del rendimiento.

## Requisitos
Para instalar todas las dependencias necesarias, asegúrate de tener un entorno de Python configurado y luego ejecuta:

```bash
pip install -r requerimientosFR.txt

```
## Demostracion de interfaz
![image](https://github.com/user-attachments/assets/8852148e-d27f-413b-bec8-26b1fe7b0610)

