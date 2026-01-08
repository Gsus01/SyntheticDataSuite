# 3. Infraestructura y Orquestación (Kubernetes)

Este apartado detalla cómo se estructura la infraestructura técnica sobre la que corre la plataforma. El objetivo es explicar las decisiones de diseño a nivel de clúster Kubernetes: cómo se organizan los recursos, cómo se exponen los servicios al exterior, y cómo se gestionan las configuraciones y los datos.

En este capítulo describimos la **arquitectura objetivo** pensada para un entorno de producción. Al final de cada sección se incluye una nota sobre el estado del prototipo actual, que simplifica ciertos aspectos para facilitar la validación técnica.

---

## 3.1. Diseño del Clúster

### Topología General

Un clúster de Kubernetes se compone de dos tipos de nodos:

- **Nodos del Control Plane (anteriormente llamados "Master")**: Son los cerebros del clúster. Ejecutan los componentes que toman decisiones sobre el sistema (por ejemplo, en qué nodo se despliega cada pod), mantienen el estado deseado y exponen la API de Kubernetes. Los componentes principales son:
  - **API Server**: El punto de entrada para todas las operaciones del clúster.
  - **etcd**: Base de datos distribuida donde se guarda todo el estado del clúster.
  - **Scheduler**: Decide en qué nodo físico se ejecuta cada pod.
  - **Controller Manager**: Se encarga de que el estado real coincida con el estado deseado (por ejemplo, si se piden 3 réplicas, asegura que haya 3 pods corriendo).

- **Nodos Worker**: Son los que hacen el trabajo real. Ejecutan los contenedores de las aplicaciones (los pods). Cuantos más workers tengamos, más carga podemos asumir en paralelo.

Para la arquitectura objetivo con alta disponibilidad, la configuración recomendada es:
- **3 nodos de Control Plane**: Para tolerar la caída de uno sin perder el clúster.
- **N nodos Worker**: Dimensionados según la carga prevista. Como mínimo 3, para permitir que los pods se redistribuyan si cae un nodo.

El siguiente diagrama muestra la topología de referencia del clúster:

![Cluster Topology](diagrams/3.1-cluster-topology.puml)

### Proveedor de Kubernetes

La plataforma está diseñada para ser **agnóstica respecto al proveedor** de Kubernetes. Los manifiestos (ficheros YAML) que definen los despliegues utilizan recursos estándar de la API de Kubernetes, sin dependencias de características específicas de ningún proveedor.

Esto significa que la plataforma puede desplegarse sobre:

| Proveedor | Descripción | Consideraciones |
|:----------|:------------|:----------------|
| **AWS EKS** | Servicio gestionado de Amazon | Buena integración con IAM, EBS para volúmenes, S3 nativo |
| **Azure AKS** | Servicio gestionado de Microsoft | Integración con Azure AD y Azure Disk |
| **Google GKE** | Servicio gestionado de Google | Considerado el más maduro, excelente auto-escalado |
| **On-Premise** | Instalación en servidores propios | Mayor control, pero requiere más gestión operativa |

La elección final dependerá de factores como las políticas de infraestructura de la organización, requisitos de soberanía de datos y las capacidades del equipo de operaciones.

> **Estado del prototipo**: Utiliza Minikube, un entorno local de desarrollo con un único nodo.

---

## 3.2. Estrategia de Namespaces

Los **namespaces** son la forma que tiene Kubernetes de crear particiones lógicas dentro de un mismo clúster físico. Permiten aislar recursos, aplicar políticas de seguridad específicas y gestionar cuotas de forma independiente.

### Organización por Responsabilidad

La arquitectura objetivo organiza los componentes en namespaces según su función, siguiendo un esquema de nombrado coherente con prefijo `dtwin-` (abreviatura de "Digital Twin"):

| Namespace | Contenido | Propósito |
|:----------|:----------|:----------|
| `dtwin-platform` | Backend API, Frontend | Componentes principales de la aplicación |
| `dtwin-workflows` | Argo Server, Argo Controller, Workflow Pods | Motor de ejecución de flujos de trabajo |
| `dtwin-storage` | MinIO (o S3) | Almacenamiento de artefactos y resultados |
| `dtwin-data` | PostgreSQL | Base de datos de metadatos (usuarios, histórico) |
| `dtwin-infra` | Ingress Controller, Cert Manager | Infraestructura de plataforma |

Esta separación aporta varias ventajas:

- **Claridad organizativa**: Cada equipo sabe dónde buscar sus recursos.
- **Aislamiento de fallos**: Un error en el namespace de workflows no afecta directamente al frontend.
- **RBAC granular**: Se pueden dar permisos a desarrolladores solo sobre `dtwin-platform`, mientras que operaciones tiene acceso a todo.
- **Cuotas de recursos**: Cada namespace puede tener límites de CPU y memoria independientes.

### Aislamiento por Entorno

Además de la separación funcional, en producción se recomienda replicar la estructura para cada entorno del ciclo de vida:

```
dtwin-platform-dev      # Desarrollo
dtwin-platform-staging  # Pre-producción
dtwin-platform-prod     # Producción
```

Con Network Policies de Kubernetes se puede garantizar que los entornos estén completamente aislados entre sí.

> **Estado del prototipo**: Utiliza namespaces con nombres simplificados (`syntheticdata`, `argo`, `minio-dev`) sin separación por entorno.

---

## 3.3. Recursos y Cargas de Trabajo

Esta sección describe los distintos tipos de objetos de Kubernetes que se utilizan para desplegar los componentes de la plataforma.

### Deployments

Los componentes de larga duración (backend y frontend) se despliegan como **Deployments**. Un Deployment gestiona aplicaciones stateless (sin estado persistente en el pod):

- Define cuántas réplicas del pod deben estar corriendo.
- Gestiona las actualizaciones de forma gradual (rolling updates).
- Si un pod falla, automáticamente crea uno nuevo para reemplazarlo.

Configuración recomendada para el backend:

```yaml
spec:
  replicas: 3                    # Alta disponibilidad
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1                # Crear 1 pod nuevo antes de eliminar viejos
      maxUnavailable: 0          # Nunca tener menos de 3 pods
  template:
    spec:
      containers:
        - name: backend
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
```

### Health Checks

Los pods tienen configuradas sondas de salud:

- **Liveness Probe**: Comprueba si el pod sigue vivo. Si falla varias veces, Kubernetes reinicia el contenedor.
- **Readiness Probe**: Comprueba si está listo para recibir tráfico. Si falla, el pod se saca temporalmente del balanceador.
- **Startup Probe** (recomendada): Da tiempo extra a aplicaciones que tardan en arrancar, evitando reinicios prematuros.

### Pods Efímeros de Workflows

Los pasos individuales de un workflow no son Deployments permanentes. Son **pods efímeros** que Argo crea y destruye dinámicamente:

1. El usuario lanza un workflow desde la UI.
2. Argo lee el DAG y determina qué pasos pueden ejecutarse.
3. Para cada paso listo, crea un pod con la imagen Docker del componente.
4. El pod ejecuta su tarea, escribe los outputs y termina.
5. Argo detecta la finalización y lanza los siguientes pasos dependientes.

Este modelo tiene una ventaja clave: **los recursos solo se consumen mientras hay trabajo**. No hay contenedores ociosos.

### StatefulSets

Los **StatefulSets** son para aplicaciones con estado que necesitan identidad estable y almacenamiento persistente. En la arquitectura objetivo se usan para:

| Componente | Por qué StatefulSet |
|:-----------|:--------------------|
| PostgreSQL | Cada réplica necesita sus propios datos. El primario se distingue de las réplicas. |
| MinIO (modo distribuido) | Cada nodo del cluster MinIO tiene su propio volumen persistente. |

### DaemonSets

Los **DaemonSets** garantizan que un pod específico corra en cada nodo del clúster. En la arquitectura objetivo:

| Agente | Propósito |
|:-------|:----------|
| Promtail / Fluentd | Recolección de logs de todos los nodos |
| Node Exporter | Métricas del sistema operativo (CPU, memoria, disco) |

> **Estado del prototipo**: Usa Deployments para backend/frontend (1 réplica). No hay StatefulSets ni DaemonSets. MinIO y los servicios externos están en configuración mínima.

---

## 3.4. Red y Exposición de Servicios

### Services

Los pods son efímeros y sus IPs cambian constantemente. Los **Services** proporcionan una abstracción estable:

- Un nombre DNS fijo (por ejemplo, `backend-svc.dtwin-platform.svc.cluster.local`).
- Una IP virtual que no cambia.
- Balanceo de carga automático entre réplicas.

| Tipo de Service | Cuándo usarlo |
|:----------------|:--------------|
| **ClusterIP** | Comunicación interna entre componentes |
| **LoadBalancer** | Exponer el Ingress Controller al exterior (lo aprovisiona el cloud provider) |

### Ingress: Arquitectura de Subdominios

La arquitectura objetivo utiliza un modelo de **subdominios** para exponer los diferentes servicios. Esto ofrece ventajas sobre el modelo de rutas (paths):

- Separación clara de responsabilidades.
- Posibilidad de aplicar políticas de seguridad diferentes por servicio.
- Más flexibilidad para escalar o migrar componentes individualmente.

| Subdominio | Servicio | Propósito |
|:-----------|:---------|:----------|
| `app.dtwin.io` | Frontend | Interfaz visual de usuario |
| `api.dtwin.io` | Backend API | Endpoints REST para la aplicación |
| `workflows.dtwin.io` | Argo Server UI | Monitorización de ejecuciones (acceso restringido) |

El siguiente diagrama muestra el flujo de tráfico:

![Ingress Routing](diagrams/3.4-ingress-routing.puml)

### Componentes de la Capa de Red

| Componente | Función |
|:-----------|:--------|
| **DNS externo** | Resuelve `*.dtwin.io` a la IP del Load Balancer |
| **Cloud Load Balancer** | Termina SSL, protege contra DDoS, balancea entre nodos |
| **Ingress Controller (Nginx)** | Enruta según host/path, aplica rate limiting, inyecta headers |
| **Cert Manager** | Provisiona certificados SSL automáticamente via Let's Encrypt |

### Seguridad en la Capa de Red

Configuraciones recomendadas en el Ingress:

- **TLS obligatorio**: Redirigir HTTP a HTTPS.
- **Rate limiting**: Proteger la API contra abusos.
- **CORS configurado**: Permitir solo orígenes conocidos.
- **Headers de seguridad**: HSTS, X-Content-Type-Options, etc.

> **Estado del prototipo**: Usa path-based routing (`/api/*`) con Nginx Ingress addon de Minikube. Sin SSL, sin rate limiting.

---

## 3.5. Almacenamiento Persistente

### Conceptos Clave

- **StorageClass**: Define qué tipo de almacenamiento está disponible (SSD, HDD, NFS). Es el "catálogo" de opciones.
- **PersistentVolume (PV)**: Un trozo concreto de almacenamiento aprovisionado.
- **PersistentVolumeClaim (PVC)**: Una solicitud de almacenamiento por parte de una aplicación.

### Arquitectura de Almacenamiento

La plataforma tiene dos tipos principales de datos persistentes:

| Tipo de Dato | Almacenamiento | Tecnología |
|:-------------|:---------------|:-----------|
| **Artefactos de workflows** | Object Storage | MinIO (S3-compatible) |
| **Metadatos y estado** | Base de datos relacional | PostgreSQL |

El flujo de datos a través del almacenamiento:

![Storage Architecture](diagrams/3.5-storage-architecture.puml)

### MinIO: Almacenamiento de Objetos

MinIO almacena todos los ficheros que fluyen entre pasos del workflow:

- Datasets de entrada.
- Resultados intermedios (CSVs procesados, modelos entrenados).
- Outputs finales.
- Logs de ejecución.

Configuración recomendada para producción:

| Aspecto | Configuración |
|:--------|:--------------|
| **Modo** | Distribuido (3+ nodos) para redundancia |
| **StorageClass** | SSD para rendimiento |
| **Capacidad** | Dimensionada según volumen de datos esperado |
| **Retención** | Política para limpiar artefactos antiguos (ej: 30 días) |

### PostgreSQL: Metadatos

PostgreSQL almacena información estructurada:

- Usuarios y permisos.
- Definiciones de workflows guardados.
- Histórico de ejecuciones.
- Configuración de componentes.

Configuración recomendada:

| Aspecto | Configuración |
|:--------|:--------------|
| **Despliegue** | StatefulSet con Primary + Replica |
| **StorageClass** | SSD con snapshots habilitados |
| **Backups** | pg_dump diario a Object Storage externo |

### Estrategia de Backup

Todos los datos críticos deben tener respaldo:

- **MinIO**: Replicación síncrona entre nodos + backup asíncrono a S3/Azure Blob externo.
- **PostgreSQL**: WAL archiving + pg_dump diario.
- **Retención**: 30 días de backups incrementales, 12 meses de snapshots mensuales.

> **Estado del prototipo**: MinIO usa HostPath (directorio local). No hay PostgreSQL ni backups automatizados.

---

## 3.6. Gestión de Configuración y Secretos

### ConfigMaps

Los **ConfigMaps** almacenan configuración no sensible en pares clave-valor:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: backend-config
  namespace: dtwin-platform
data:
  STORAGE_ENDPOINT: "minio.dtwin-storage.svc.cluster.local:9000"
  ARGO_SERVER_URL: "argo-server.dtwin-workflows.svc.cluster.local:2746"
  LOG_LEVEL: "INFO"
```

Ventajas:
- Cambiar configuración sin modificar imágenes Docker.
- Diferentes valores por entorno (dev/staging/prod).

### Secrets

Los **Secrets** almacenan información sensible (contraseñas, tokens, claves API). En Kubernetes estándar están codificados en Base64, pero **no cifrados**.

Para la arquitectura objetivo se recomienda:

1. **Cifrado en reposo**: Habilitar Encryption at Rest en etcd.
2. **Gestor de secretos externo**: Integrar con HashiCorp Vault, AWS Secrets Manager o Azure Key Vault.
3. **Rotación automática**: Los secretos deben poder rotarse sin reiniciar pods.

### Organización de Secretos

| Secret | Contenido | Namespace |
|:-------|:----------|:----------|
| `db-credentials` | Usuario/contraseña PostgreSQL | `dtwin-data` |
| `minio-credentials` | Access Key / Secret Key | `dtwin-storage` |
| `oidc-client` | Client ID/Secret para OAuth2 | `dtwin-platform` |

> **Estado del prototipo**: Usa Secrets básicos de Kubernetes con credenciales por defecto.

---

## Resumen Comparativo

| Aspecto | Arquitectura Objetivo | Prototipo Actual |
|:--------|:----------------------|:-----------------|
| **Clúster** | Multi-nodo con HA (3 masters, N workers) | Minikube (1 nodo) |
| **Proveedor** | AWS EKS / Azure AKS / On-prem | Minikube local |
| **Namespaces** | `dtwin-platform`, `dtwin-workflows`, `dtwin-storage`, `dtwin-data`, `dtwin-infra` | `syntheticdata`, `argo`, `minio-dev` |
| **Backend/Frontend** | 3+ réplicas con auto-scaling | 1 réplica fija |
| **Base de datos** | PostgreSQL en StatefulSet | No implementado |
| **Object Storage** | MinIO distribuido con replicación | MinIO standalone con HostPath |
| **Ingress** | Subdominios + SSL + rate limiting | Path-based sin SSL |
| **Secrets** | Vault + cifrado etcd | Secrets básicos K8s |
| **Observabilidad** | Prometheus + Grafana + DaemonSets | Logs nativos + UI Argo |

El prototipo actual demuestra que la arquitectura funciona a nivel técnico. La evolución hacia producción implica escalar (más réplicas, más nodos), añadir las capas de seguridad y observabilidad, y adoptar las prácticas de gestión de secretos descritas.
