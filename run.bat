@echo off
REM Script simple para ejecutar el bot en Windows

echo 🤖 Iniciando Bot de Señales de Trading...
echo.

REM Verificar que existe el archivo .env
if not exist .env (
    echo ❌ Error: No se encontró el archivo .env
    echo.
    echo Por favor:
    echo 1. Copia el archivo .env.example a .env
    echo 2. Edita .env con tu TOKEN de Telegram y CHAT_ID
    echo.
    echo Comando: copy .env.example .env
    pause
    exit /b 1
)

echo ✓ Archivo .env encontrado
echo.

REM Activar entorno virtual si existe
if exist venv\Scripts\activate.bat (
    echo ✓ Activando entorno virtual...
    call venv\Scripts\activate.bat
)

REM Ejecutar el bot
echo 🚀 Iniciando bot...
echo.
python main.py

pause
