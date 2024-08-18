# Utiliza una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Instala las dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Copia el archivo requirements.txt en el directorio de trabajo
COPY app/requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación en el directorio de trabajo
COPY app/ .

# Copia los modelos y la carpeta de media en el directorio de trabajo
COPY modelos ./modelos
COPY media ./media

# Expone el puerto en el que correrá la aplicación
EXPOSE 8080

# Comando para correr la aplicación con Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]