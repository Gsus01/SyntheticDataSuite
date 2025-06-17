#!/bin/bash

# Script de ayuda para usar el preprocesador de series temporales con Docker Compose

echo "🚀 PREPROCESADOR DE SERIES TEMPORALES - GUÍA DE USO"
echo "=" | tr ' ' '=' | head -c 60; echo

echo "📋 COMANDOS RÁPIDOS DISPONIBLES:"
echo

echo "1. 🔬 Test rápido (datos de sensores):"
echo "   docker compose run --rm test-preprocessor"
echo

echo "2. 🌡️  Procesar datos de sensores:"
echo "   docker compose run --rm process-sensors"
echo

echo "3. 💰 Procesar datos financieros:"
echo "   docker compose run --rm process-financial"
echo

echo "4. ⏰ Procesar datos con timestamps irregulares:"
echo "   docker compose run --rm process-irregular"
echo

echo "5. 🔀 Procesar datos de tipos mixtos:"
echo "   docker compose run --rm process-mixed"
echo

echo "6. 📦 Procesamiento en lote:"
echo "   docker compose run --rm batch-processor"
echo

echo "7. 🎯 Comando personalizado:"
echo "   docker compose run --rm test-preprocessor \\"
echo "     --config /app/config.yaml \\"
echo "     --input /app/data/input/TU_ARCHIVO.csv \\"
echo "     --output /app/data/output/TU_SALIDA.csv"
echo

echo "📁 DIRECTORIOS:"
echo "   • Entrada: ./test_data/"
echo "   • Salida:  ./output/"
echo "   • Config:  ./config.yaml"
echo

echo "🔧 COMANDOS ÚTILES:"
echo "   • Construir imagen:     docker compose build"
echo "   • Ver archivos salida:  ls -la output/"
echo "   • Limpiar salida:       rm -rf output/* && mkdir -p output"
echo

echo "💡 EJEMPLOS DE USO:"
echo "   # Test rápido"
echo "   docker compose run --rm test-preprocessor"
echo
echo "   # Procesar archivo específico"
echo "   docker compose run --rm test-preprocessor \\"
echo "     --input /app/data/input/sensor_data_with_nulls.csv \\"
echo "     --output /app/data/output/mi_resultado.csv"
echo

echo "📊 VERIFICAR RESULTADOS:"
echo "   head -5 output/processed_*.csv"
echo "   wc -l output/*.csv"
echo
