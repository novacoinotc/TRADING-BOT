# Usar imagen oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno para evitar problemas de caché
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copiar archivos de requerimientos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# CRÍTICO: Limpiar todo el caché de Python para evitar ejecutar código viejo
RUN find . -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    echo "✅ Caché de Python limpiado"

# Crear directorio de logs
RUN mkdir -p logs data

# Comando para ejecutar el bot
CMD ["python", "-B", "main.py"]
