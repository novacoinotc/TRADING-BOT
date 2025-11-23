#!/bin/bash
echo "🧹 Limpiando caché de Python..."

# Eliminar todos los archivos .pyc
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Archivos .pyc eliminados"

# Eliminar directorios __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✅ Directorios __pycache__ eliminados"

# Eliminar caché de pip
rm -rf ~/.cache/pip 2>/dev/null
echo "✅ Caché de pip limpiado"

# Eliminar .pytest_cache si existe
rm -rf .pytest_cache 2>/dev/null

echo "🎉 Caché limpiado completamente"
