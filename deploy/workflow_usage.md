# Synthetic Data Generation Workflow

Este documento describe el flujo de trabajo de Argo para la generación de datos sintéticos utilizando múltiples modelos de aprendizaje automático.

## Arquitectura del Workflow

El workflow está diseñado como un DAG (Directed Acyclic Graph) con las siguientes fases:

### 1. Fase de Preprocesamiento
- **Template**: `preprocessing`
- **Imagen**: `docker.io/library/preprocessing:v2`
- **Función**: Procesa y limpia los datos de entrada
- **Dependencias**: Ninguna (punto de entrada)

### 2. Fase de Entrenamiento (Paralelo)
Todos los modelos se entrenan simultáneamente después del preprocesamiento:

- **HMM (Hidden Markov Models)**
  - Template: `train-hmm-model`
  - Imagen: `docker.io/library/training-hmm:v2`

- **Gaussian Process**
  - Template: `train-gaussian-process-model`
  - Imagen: `docker.io/library/training-gaussian_process:v2`

- **Copulas**
  - Template: `train-copulas-model`
  - Imagen: `docker.io/library/training-copulas:v2`

- **Boltzmann Machines**
  - Template: `train-boltzman-machines-model`
  - Imagen: `docker.io/library/training-boltzman_machines:v2`

- **Bayesian Networks**
  - Template: `train-bayesian-networks-model`
  - Imagen: `docker.io/library/training-bayesian_networks:v2`

### 3. Fase de Generación (Paralelo)
Cada modelo entrenado genera datos sintéticos independientemente:

- **HMM Data Generation**
  - Template: `generate-hmm-data`
  - Imagen: `docker.io/library/generation-hmm:v2`

- **Gaussian Process Data Generation**
  - Template: `generate-gaussian-process-data`
  - Imagen: `docker.io/library/generation-gaussian_process:v2`

- **Copulas Data Generation**
  - Template: `generate-copulas-data`
  - Imagen: `docker.io/library/generation-copulas:v2`

- **Boltzmann Machines Data Generation**
  - Template: `generate-boltzman-machines-data`
  - Imagen: `docker.io/library/generation-boltzman_machines:v2`

- **Bayesian Networks Data Generation**
  - Template: `generate-bayesian-networks-data`
  - Imagen: `docker.io/library/generation-bayesian_networks:v2`

## Manejo de Artefactos

### Configuración de Archivos
- **Sin Compresión**: Todos los artefactos están configurados con `archive: none: {}` para transferencia directa
- **Archivos Directos**: Los CSV se transfieren como CSV, PKL como PKL (sin empaquetado TAR)

### Tipos de Entrada Soportados
1. **Archivos Locales**: Se suben automáticamente al repositorio de artefactos
2. **URLs S3/MinIO**: Referencias directas a archivos ya almacenados
3. **ConfigMaps**: Para archivos pequeños de configuración

### Flujo de Artefactos
1. **Entrada**: Datos originales y archivos de configuración
2. **Preprocesamiento**: Genera datos procesados
3. **Entrenamiento**: Cada modelo consume datos procesados y genera modelos entrenados
4. **Generación**: Cada modelo entrenado genera datos sintéticos

## Uso del Workflow

### Prerequisitos
- Cluster de Kubernetes con Argo Workflows instalado
- Repositorio de artefactos configurado (S3, MinIO, etc.)
- Imágenes Docker disponibles en el registro especificado

### Ejecución

#### Opción 1: Workflow Simplificado (Archivos Incluidos en Imágenes)
```bash
# Ejecución directa - los archivos están incluidos en las imágenes Docker
argo submit deploy/general_workflow.yaml

# Con logs en tiempo real
argo submit deploy/general_workflow.yaml --log
```

#### Opción 2: Subir Archivos a MinIO Manualmente
```bash
# 1. Hacer port-forward a MinIO (en terminal separada)
kubectl -n minio-dev port-forward pod/minio 9090:9090 9000:9000

# 2. Subir archivos via consola web (http://localhost:9090)
# Usuario: minioadmin, Password: minioadmin

# 3. Ejecutar workflow
argo submit deploy/general_workflow.yaml
```

#### Opción 3: Usando AWS CLI con MinIO
```bash
# Configurar AWS CLI para MinIO local
aws configure set aws_access_key_id minioadmin
aws configure set aws_secret_access_key minioadmin

# Subir archivos
aws --endpoint-url http://localhost:9000 s3 cp ./data/my_data.csv s3://argo-artifacts/input/
aws --endpoint-url http://localhost:9000 s3 cp ./config/my_config.yaml s3://argo-artifacts/config/

# Ejecutar workflow
argo submit deploy/general_workflow.yaml
```

### Parámetros de Entrada
El workflow simplificado no requiere parámetros externos:
- Los archivos de datos y configuración están incluidos en las imágenes Docker
- Para usar archivos personalizados, súbelos a MinIO primero

### Limitaciones y Consideraciones

| Método | Ventajas | Desventajas | Uso Recomendado |
|--------|----------|-------------|-----------------|
| **Archivos en Imágenes** | ✅ Muy fácil<br>✅ Sin configuración<br>✅ Funciona inmediatamente | ❌ Archivos fijos<br>❌ Imágenes más grandes | Testing rápido |
| **MinIO Manual** | ✅ Archivos personalizados<br>✅ Flexible<br>✅ Buen rendimiento | ❌ Pasos manuales<br>❌ Requiere MinIO configurado | Desarrollo |
| **AWS CLI + MinIO** | ✅ Scripteable<br>✅ Automático<br>✅ CI/CD friendly | ❌ Configuración inicial<br>❌ Dependencia AWS CLI | Producción |

### Artefactos de Salida
El workflow genera los siguientes artefactos:

1. **Datos Procesados**: `/tmp/output/processed_data.csv`
2. **Modelos Entrenados**:
   - HMM: `/tmp/output/hmm_model.pkl`
   - Gaussian Process: `/tmp/output/gp_model.pkl`
   - Copulas: `/tmp/output/copulas_model.pkl`
   - Boltzmann Machines: `/tmp/output/rbm_model.pkl`
   - Bayesian Networks: `/tmp/output/bn_model.pkl`
3. **Datos Sintéticos**: Cada modelo genera `/tmp/output/synthetic_data.csv`

## Monitoreo y Debugging

### Ver Estado del Workflow
```bash
# Listar workflows
argo list

# Ver detalles de un workflow específico
argo get <workflow-name>

# Ver logs de un paso específico
argo logs <workflow-name> -c <container-name>
```

### Troubleshooting Común

1. **Errores de Artefactos**: Verificar configuración del repositorio de artefactos
2. **Imágenes no Encontradas**: Asegurar que las imágenes Docker estén disponibles
3. **Permisos**: Verificar que el ServiceAccount tenga los permisos necesarios
4. **Archivos Locales no Encontrados**: 
   ```bash
   # Verificar que el archivo existe
   ls -la ./data/my_data.csv
   
   # Usar rutas absolutas si hay problemas
   argo submit general_workflow.yaml \
     --artifact input-data=$(pwd)/data/my_data.csv \
     --artifact config-file=$(pwd)/config/my_config.yaml
   ```
5. **Problemas con S3/MinIO URLs**:
   ```bash
   # Verificar conectividad
   aws s3 ls s3://my-bucket/data/
   
   # Verificar credenciales
   kubectl get secret -n argo
   ```

### Diferencias Entre Métodos de Entrada

| Método | Ventajas | Desventajas | Uso Recomendado |
|--------|----------|-------------|-----------------|
| **Archivos Locales** | ✅ Fácil de usar<br>✅ No requiere S3 configurado<br>✅ Funciona offline | ❌ Upload inicial más lento | Desarrollo y testing |
| **URLs S3/MinIO** | ✅ Muy rápido<br>✅ Archivos ya almacenados<br>✅ Ideal para producción | ❌ Requiere configurar S3<br>❌ Archivos deben estar subidos | Producción |
| **ConfigMaps** | ✅ Nativo de Kubernetes<br>✅ Versionado con Git | ❌ Limitado a 1MB<br>❌ Solo para configs | Configs pequeños |

## Optimización de Rendimiento

### Paralelización
- El workflow está optimizado para ejecutar entrenamiento en paralelo
- La generación de datos también se ejecuta en paralelo para cada modelo

### Recursos
- Configurar límites de recursos apropiados para cada contenedor
- Considerar el uso de `podSpecPatch` para artefactos grandes

```yaml
podSpecPatch: |
  initContainers:
    - name: init
      resources:
        requests:
          memory: 2Gi
          cpu: 300m
```

## Extensión del Workflow

Para añadir nuevos modelos:

1. Crear las imágenes Docker correspondientes
2. Añadir templates de entrenamiento y generación
3. Actualizar el DAG con las nuevas dependencias
4. Configurar el manejo de artefactos apropiado

## Referencias

- [Documentación de Argo Workflows](https://argo-workflows.readthedocs.io/)
- [Manejo de Artefactos en Argo](https://argo-workflows.readthedocs.io/en/latest/walk-through/artifacts/)
- [DAG Templates](https://argo-workflows.readthedocs.io/en/latest/walk-through/dag/)
