#!/bin/bash

# Script simple para ejecutar el bot de señales de trading

echo "🤖 Iniciando Bot de Señales de Trading..."
echo ""

# Verificar que existe el archivo .env
if [ ! -f .env ]; then
    echo "❌ Error: No se encontró el archivo .env"
    echo ""
    echo "Por favor:"
    echo "1. Copia el archivo .env.example a .env"
    echo "2. Edita .env con tu TOKEN de Telegram y CHAT_ID"
    echo ""
    echo "Comando: cp .env.example .env"
    exit 1
fi

# Verificar Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python no está instalado"
    echo "Por favor instala Python 3.8 o superior"
    exit 1
fi

echo "✓ Python encontrado: $PYTHON"
echo "✓ Archivo .env encontrado"
echo ""

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "✓ Activando entorno virtual..."
    source venv/bin/activate
fi

# Ejecutar el bot
echo "🚀 Iniciando bot..."
echo ""
$PYTHON main.py
