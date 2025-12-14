# Usamos una imagen base de Python ligera (Linux)
FROM python:3.11-slim

# Evitar que Python genere archivos .pyc y buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo en el "contenedor"
WORKDIR /app

# Instalar dependencias del sistema necesarias para ChromaDB
# (Compiladores de C++ básicos)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copiar primero el fichero de requisitos (para aprovechar caché de Docker)
COPY requirements.txt .

# 2. Instalar librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para arrancar la app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
