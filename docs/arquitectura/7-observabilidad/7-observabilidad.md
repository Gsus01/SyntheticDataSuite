# 7. Observabilidad y Operaciones

## Introducción

La observabilidad es la capacidad de comprender el estado interno de un sistema a partir de sus salidas externas: logs, métricas y trazas. En una plataforma distribuida sobre Kubernetes, esta capacidad es esencial para detectar problemas, diagnosticar incidencias y validar que los workflows se ejecutan correctamente.

Este apartado describe la estrategia de observabilidad de la arquitectura objetivo, organizada en cuatro pilares: logging, métricas, tracing y alertas.

---

## 7.1. Logging

### Objetivo

Centralizar los logs de todos los componentes (backend, frontend, pods de workflow, servicios de infraestructura) en un sistema que permita búsqueda, filtrado y correlación temporal.

### Stack Recomendado

La arquitectura contempla **Loki + Grafana** como solución de logging, por su integración nativa con el ecosistema Prometheus/Grafana y su menor consumo de recursos frente a alternativas como Elasticsearch.

| Componente | Función |
|:-----------|:--------|
| **Promtail** | Agente que recolecta logs de los pods y los envía a Loki |
| **Loki** | Backend de almacenamiento indexado por etiquetas (no por contenido) |
| **Grafana** | Interfaz de consulta y visualización |

### Alternativas

- **ELK (Elasticsearch + Logstash + Kibana)**: más potente en búsqueda full-text, pero mayor complejidad operativa y consumo de recursos.
- **EFK (Elasticsearch + Fluentd + Kibana)**: variante con Fluentd como colector, común en entornos Kubernetes.

Para la mayoría de casos de uso de la plataforma, Loki es suficiente y se alinea mejor con una filosofía "cloud-native" ligera.

### Estrategia de Etiquetado

Los logs deben incluir etiquetas que permitan filtrar por:
- `namespace`: entorno (dev, pre, prod)
- `app`: nombre del servicio (backend, frontend, workflow-step)
- `workflow`: nombre del workflow en ejecución
- `node`: identificador del paso del workflow

### Estado del Prototipo

El prototipo no implementa agregación de logs. Los logs se consultan directamente con `kubectl logs` o a través de la interfaz de Argo Workflows. Para producción, se recomienda desplegar Loki como parte del stack de observabilidad.

---

## 7.2. Monitorización y Métricas

### Objetivo

Recopilar métricas de infraestructura (CPU, memoria, red) y de negocio (workflows ejecutados, tiempos, tasas de error) para detectar degradaciones y dimensionar recursos.

### Stack Recomendado

Se adopta **Prometheus + Grafana**, el estándar de facto en Kubernetes.

| Componente | Función |
|:-----------|:--------|
| **Prometheus** | Recolección y almacenamiento de métricas en formato time-series |
| **Node Exporter** | Métricas de nodo (CPU, memoria, disco) |
| **kube-state-metrics** | Métricas del estado de objetos Kubernetes (pods, deployments) |
| **Grafana** | Dashboards y visualización |

### Tipos de Métricas

**Métricas de infraestructura** (automáticas con kube-prometheus-stack):
- CPU y memoria por pod/contenedor
- Requests y limits utilizados
- Estado de pods (Running, Pending, Failed)
- Latencia de red

**Métricas de aplicación** (requieren instrumentación):
- Requests HTTP por endpoint (latencia p50, p95, p99)
- Workflows ejecutados por estado (succeeded, failed)
- Tiempos de ejecución por paso de workflow
- Artefactos procesados (tamaño, cantidad)

### Instrumentación

El backend (FastAPI) puede exponer métricas en formato Prometheus mediante librerías como `prometheus-fastapi-instrumentator`. Argo Workflows también expone métricas nativas que Prometheus puede scrapear.

### Estado del Prototipo

El prototipo no incluye Prometheus ni Grafana. Las métricas se observan manualmente a través de `kubectl top` y la consola de Argo. Para producción, se recomienda el stack kube-prometheus-stack (Helm chart).

---

## 7.3. Tracing Distribuido

### Objetivo

Seguir el recorrido de una petición o workflow a través de múltiples servicios, identificando cuellos de botella y puntos de fallo.

### Cuándo es Necesario

El tracing distribuido aporta valor cuando:
- Existen múltiples servicios que participan en una misma operación.
- Se necesita diagnosticar latencias en flujos complejos.
- Se requiere correlacionar logs de diferentes componentes.

En la arquitectura objetivo con servicios desacoplados, el tracing sería útil. En el prototipo monolítico, su valor es limitado.

### Stack Recomendado

**Jaeger** o **Tempo** (Grafana) para almacenamiento de trazas, con instrumentación OpenTelemetry.

| Componente | Función |
|:-----------|:--------|
| **OpenTelemetry SDK** | Instrumentación de aplicaciones |
| **Jaeger / Tempo** | Backend de almacenamiento de trazas |
| **Grafana** | Visualización (integración nativa con Tempo) |

### Estado del Prototipo

No implementado. Se considera opcional para la fase de validación técnica. Puede incorporarse cuando se evolucione hacia una arquitectura de microservicios.

---

## 7.4. Gestión de Alertas

### Objetivo

Notificar proactivamente cuando se detectan condiciones anómalas, antes de que impacten al usuario o se conviertan en incidencias graves.

### Componente

**Alertmanager** (parte del stack Prometheus) gestiona el ciclo de vida de las alertas: agrupa, silencia, enruta y notifica.

### Tipos de Alertas

**Críticas** (requieren acción inmediata):
- Pod en CrashLoopBackOff durante más de 5 minutos
- Uso de CPU o memoria > 90% sostenido
- Workflow fallido con código de error inesperado
- Almacenamiento de MinIO > 85%

**Advertencias** (requieren atención):
- Latencia p99 > 2 segundos
- Tasa de error > 5% en ventana de 5 minutos
- Pods en Pending durante más de 10 minutos

**Informativas** (para dashboards):
- Nuevo release desplegado
- Workflow completado exitosamente

### Canales de Notificación

Alertmanager soporta múltiples receptores: email, Slack, PagerDuty, webhooks. La configuración depende de la política de on-call de la organización.

### Estado del Prototipo

No implementado. Para producción, Alertmanager se despliega como parte del stack kube-prometheus.

---

## Diagrama de Observabilidad

![Observability Stack](diagrams/7-observability-stack.puml)

---

## Resumen

| Pilar | Herramienta Recomendada | Estado Prototipo |
|:------|:------------------------|:-----------------|
| Logging | Loki + Promtail + Grafana | No implementado |
| Métricas | Prometheus + Grafana | No implementado |
| Tracing | Jaeger / Tempo + OpenTelemetry | No implementado |
| Alertas | Alertmanager | No implementado |

La observabilidad no forma parte del prototipo de validación técnica, pero es imprescindible para cualquier despliegue en producción. El stack recomendado (Prometheus + Loki + Grafana + Alertmanager) es maduro, bien documentado y ampliamente adoptado en entornos Kubernetes.
