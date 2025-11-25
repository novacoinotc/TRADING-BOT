#!/bin/bash

echo "╔════════════════════════════════════════════════════╗"
echo "║   🤖 INSTALADOR AUTOMÁTICO - BOT DE SEÑALES      ║"
echo "║        Para Mac - Ultra Simple                     ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Colores para mejor visualización
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar si estamos en el directorio correcto
if [ ! -f "main.py" ]; then
    echo "${RED}❌ Error: No se encontró main.py${NC}"
    echo "Por favor, ejecuta este script desde la carpeta TRADING-BOT"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Paso 1: Verificando Python..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verificar Python3
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "${GREEN}✅ Python encontrado: $PYTHON_VERSION${NC}"
    PYTHON=python3
    PIP=pip3
else
    echo "${RED}❌ Python no está instalado${NC}"
    echo ""
    echo "Por favor instala Python desde: https://www.python.org/downloads/"
    echo "O usando Homebrew: brew install python3"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Paso 2: Verificando archivo .env..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f ".env" ]; then
    echo "${YELLOW}⚠️  Archivo .env no encontrado${NC}"
    echo "Creando desde plantilla..."
    cp .env.example .env
    echo "${GREEN}✅ Archivo .env creado${NC}"
    echo ""
    echo "${YELLOW}⚠️  IMPORTANTE: Edita el archivo .env con tus credenciales:${NC}"
    echo "   - TELEGRAM_BOT_TOKEN"
    echo "   - TELEGRAM_CHAT_ID"
    echo ""
    echo "Presiona ENTER cuando hayas editado el .env..."
    read
else
    echo "${GREEN}✅ Archivo .env encontrado${NC}"
fi

# Verificar que el .env tenga las variables necesarias
if grep -q "your_bot_token_here" .env || grep -q "your_chat_id_here" .env; then
    echo ""
    echo "${RED}❌ ERROR: El archivo .env no está configurado correctamente${NC}"
    echo ""
    echo "Por favor edita el archivo .env y reemplaza:"
    echo "  - your_bot_token_here → Tu token de @BotFather"
    echo "  - your_chat_id_here → Tu Chat ID de @userinfobot"
    echo ""
    echo "Para editar el archivo, ejecuta:"
    echo "  nano .env"
    echo ""
    echo "Luego ejecuta este script de nuevo."
    exit 1
fi

echo "${GREEN}✅ Archivo .env configurado correctamente${NC}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Paso 3: Instalando dependencias..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Esto puede tomar 2-3 minutos..."
echo ""

# Actualizar pip
echo "Actualizando pip..."
$PIP install --upgrade pip --quiet

# Instalar dependencias
echo "Instalando librerías necesarias..."
$PIP install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "${GREEN}✅ Todas las dependencias instaladas correctamente${NC}"
else
    echo "${RED}❌ Error al instalar dependencias${NC}"
    echo "Intenta ejecutar manualmente:"
    echo "  pip3 install -r requirements.txt"
    exit 1
fi

# Crear directorio de logs
mkdir -p logs

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ INSTALACIÓN COMPLETADA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Para iniciar el bot, ejecuta:"
echo ""
echo "  ${GREEN}$PYTHON main.py${NC}"
echo ""
echo "O simplemente:"
echo ""
echo "  ${GREEN}./run.sh${NC}"
echo ""
echo "¿Quieres iniciar el bot ahora? (s/n)"
read -r response

if [[ "$response" =~ ^([sS][iI]|[sS])$ ]]; then
    echo ""
    echo "🚀 Iniciando bot de señales..."
    echo ""
    $PYTHON main.py
else
    echo ""
    echo "Ok. Puedes iniciarlo cuando quieras con:"
    echo "  $PYTHON main.py"
fi
