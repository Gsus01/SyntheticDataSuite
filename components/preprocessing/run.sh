#!/bin/bash

# Script de utilidad para el preprocesador de series temporales
# Facilita la ejecución con Docker y validación

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo "Preprocesador de Series Temporales - Script de Utilidad"
    echo
    echo "Uso: $0 [COMANDO] [OPCIONES]"
    echo
    echo "Comandos:"
    echo "  build         Construir la imagen Docker"
    echo "  test          Ejecutar tests"
    echo "  run           Ejecutar preprocesamiento"
    echo "  demo          Ejecutar demo con datos de prueba"
    echo "  clean         Limpiar archivos temporales"
    echo "  generate      Generar datos de prueba"
    echo
    echo "Opciones para 'run':"
    echo "  --config CONFIG    Archivo de configuración (default: config.yaml)"
    echo "  --input INPUT      Archivo CSV de entrada"
    echo "  --output OUTPUT    Archivo CSV de salida"
    echo "  --docker          Ejecutar con Docker"
    echo
    echo "Ejemplos:"
    echo "  $0 build"
    echo "  $0 test"
    echo "  $0 demo"
    echo "  $0 run --input data.csv --output processed.csv"
    echo "  $0 run --input data.csv --output processed.csv --docker"
}

# Función para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Función para construir imagen Docker
build_docker() {
    log "Construyendo imagen Docker..."
    docker build -t timeseries-preprocessor .
    log "Imagen Docker construida exitosamente"
}

# Función para ejecutar tests
run_tests() {
    log "Ejecutando tests dentro de Docker..."
    docker compose -f ../../docker-compose.yml build test-preprocessing
    docker compose -f ../../docker-compose.yml run --rm test-preprocessing
}

# Función para generar datos de prueba
generate_test_data() {
    log "Generando datos de prueba..."
    python3 generate_test_data.py
    log "Datos de prueba generados en test_data/"
}

# Función para ejecutar demo
run_demo() {
    log "Ejecutando demo del preprocesador..."
    
    # Generar datos de prueba si no existen
    if [ ! -d "test_data" ] || [ -z "$(ls -A test_data 2>/dev/null)" ]; then
        generate_test_data
    fi
    
    # Crear directorio de salida
    mkdir -p demo_output
    
    # Procesar cada archivo de prueba
    for file in test_data/*.csv; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .csv)
            output_file="demo_output/processed_${filename}.csv"
            
            log "Procesando $file -> $output_file"
            python3 preprocess.py --config config.yaml --input "$file" --output "$output_file"
        fi
    done
    
    log "Demo completado. Resultados en demo_output/"
    
    # Mostrar estadísticas
    echo
    echo "Estadísticas de procesamiento:"
    echo "=============================="
    for file in demo_output/*.csv; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            echo "$(basename "$file"): $((lines-1)) filas procesadas"
        fi
    done
}

# Función para ejecutar preprocesamiento
run_preprocessing() {
    local config="config.yaml"
    local input=""
    local output=""
    local use_docker=false
    
    # Parsear argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                config="$2"
                shift 2
                ;;
            --input)
                input="$2"
                shift 2
                ;;
            --output)
                output="$2"
                shift 2
                ;;
            --docker)
                use_docker=true
                shift
                ;;
            *)
                error "Opción desconocida: $1"
                exit 1
                ;;
        esac
    done
    
    # Validar argumentos requeridos
    if [ -z "$input" ] || [ -z "$output" ]; then
        error "Se requieren --input y --output"
        show_help
        exit 1
    fi
    
    # Validar archivos
    if [ ! -f "$config" ]; then
        error "Archivo de configuración no encontrado: $config"
        exit 1
    fi
    
    if [ ! -f "$input" ]; then
        error "Archivo de entrada no encontrado: $input"
        exit 1
    fi
    
    # Crear directorio de salida
    mkdir -p "$(dirname "$output")"
    
    if [ "$use_docker" = true ]; then
        log "Ejecutando con Docker..."
        
        # Verificar que la imagen existe
        if ! docker image inspect timeseries-preprocessor >/dev/null 2>&1; then
            log "Imagen Docker no encontrada. Construyendo..."
            build_docker
        fi
        
        # Ejecutar con Docker
        docker run --rm \
            -v "$(pwd)/$config:/app/config.yaml:ro" \
            -v "$(pwd)/$input:/app/input.csv:ro" \
            -v "$(pwd)/$(dirname "$output"):/app/output" \
            timeseries-preprocessor \
            --config /app/config.yaml \
            --input /app/input.csv \
            --output "/app/output/$(basename "$output")"
    else
        log "Ejecutando localmente..."
        python3 preprocess.py --config "$config" --input "$input" --output "$output"
    fi
    
    log "Preprocesamiento completado: $output"
}

# Función para limpiar archivos temporales
clean_files() {
    log "Limpiando archivos temporales..."
    
    # Limpiar outputs de demo y test
    rm -rf demo_output/ test_output/ output/
    
    # Limpiar archivos temporales de Python
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Limpiar logs
    rm -f *.log
    
    log "Archivos temporales eliminados"
}

# Función principal
main() {
    case "${1:-help}" in
        build)
            build_docker
            ;;
        test)
            run_tests
            ;;
        run)
            shift
            run_preprocessing "$@"
            ;;
        demo)
            run_demo
            ;;
        generate)
            generate_test_data
            ;;
        clean)
            clean_files
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Comando desconocido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Verificar que estamos en el directorio correcto
if [ ! -f "preprocess.py" ]; then
    error "Este script debe ejecutarse desde el directorio del preprocesador"
    exit 1
fi

# Ejecutar función principal
main "$@"
