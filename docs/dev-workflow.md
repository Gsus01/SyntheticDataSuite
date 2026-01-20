### Guía de Desarrollo: levantar y trabajar con la suite

Este documento resume cómo arrancar el entorno de desarrollo local de forma rápida usando los scripts y el Makefile. La idea es minimizar pasos manuales y permitir arrancar sólo las piezas que necesites (MinIO, Argo, backend, frontend).

#### Requisitos

Antes de empezar, asegúrate de tener instalados:

| Herramienta | Descripción | Instalación |
|-------------|-------------|-------------|
| Docker | Motor de contenedores | [Documentación oficial](https://docs.docker.com/engine/install/) |
| minikube | Clúster Kubernetes local | [Documentación oficial](https://minikube.sigs.k8s.io/docs/start/) |
| kubectl | CLI de Kubernetes | [Documentación oficial](https://kubernetes.io/docs/tasks/tools/) |
| Argo CLI | Cliente de línea de comandos para Argo Workflows | [Releases en GitHub](https://github.com/argoproj/argo-workflows/releases) |
| Node 18+ y npm | Entorno JavaScript para el frontend | [Documentación oficial](https://nodejs.org/) |
| Python 3.10+ | Lenguaje para el backend | [Documentación oficial](https://www.python.org/downloads/) |
| uv | Gestor de dependencias Python | [Documentación oficial](https://docs.astral.sh/uv/getting-started/installation/) |

> **Nota:** Los scripts despliegan Argo Server dentro de minikube automáticamente, pero el backend necesita el binario `argo` instalado localmente para ejecutar `argo submit`.

#### Comando rápido (todo)
```bash
make dev
```
Arranca: MinIO + Argo + Postgres en minikube, abre los puertos (9090/9000 para MinIO, 2746 para Argo, 5432 para DB) y lanza frontend (3000) y backend (8000).

---

### Scripts disponibles

Todos los scripts están en `scripts/dev/`.

1) `k8s-up.sh` — despliegue en Kubernetes (minikube)
- Uso básico: `./scripts/dev/k8s-up.sh`
- Sólo un componente:
  - `./scripts/dev/k8s-up.sh --only minio`
  - `./scripts/dev/k8s-up.sh --only argo`
  - `./scripts/dev/k8s-up.sh --only db`
- Comportamiento:
  - Arranca minikube si no está corriendo (perfil configurable con `MINIKUBE_PROFILE`)
  - Aplica manifests de MinIO, Argo y Postgres (DB)
  - Ajusta Argo para `--auth-mode=server` (modo dev)
  - Espera a que los deployments estén listos
  - Si `kubectl` no existe en el PATH, usa automáticamente `minikube kubectl --`

2) `k8s-down.sh` — limpieza de namespaces
- Uso básico: `./scripts/dev/k8s-down.sh`
- Sólo un componente:
  - `./scripts/dev/k8s-down.sh --only minio`
  - `./scripts/dev/k8s-down.sh --only argo`
  - `./scripts/dev/k8s-down.sh --only db`

3) `port-forward.sh` — gestionar port-forwards y estados
- Estado: `./scripts/dev/port-forward.sh status`
- Iniciar:
  - `./scripts/dev/port-forward.sh start`
  - Selectivo: `./scripts/dev/port-forward.sh start --only argo`
  - Selectivo: `./scripts/dev/port-forward.sh start --only db`
- Parar:
  - `./scripts/dev/port-forward.sh stop`
  - Selectivo: `./scripts/dev/port-forward.sh stop --only minio`
  - Selectivo: `./scripts/dev/port-forward.sh stop --only db`
- Guarda PIDs y logs en `.tmp/dev/`
- Usa el mismo fallback a `minikube kubectl --` si `kubectl` no está disponible directamente

4) `start-stack.sh` — orquestador de todo el stack
- Todo: `./scripts/dev/start-stack.sh`
- Flags útiles:
  - `--skip-k8s` (no toca Kubernetes)
  - `--skip-pf` (no abre port-forwards)
  - `--skip-backend` (no lanza backend)
  - `--skip-frontend` (no lanza frontend)
  - `--only-k8s` (equivale a sólo Kubernetes)
  - `--only-backend`
  - `--only-frontend`
- Se cierra con `Ctrl+C` (hace cleanup de port-forwards)
- Arranca el backend con `uvicorn main:app` estableciendo `PYTHONPATH` hacia `backend/`
- Exporta `DATABASE_URL` y `RUN_MIGRATIONS` para el registry de componentes

---

### Makefile (atajos)

```bash
make dev                 # orquesta todo
make k8s-up              # sube MinIO + Argo + DB
make k8s-up-minio        # sólo MinIO
make k8s-up-argo         # sólo Argo
make k8s-up-db           # sólo DB
make k8s-down            # elimina namespaces (MinIO/Argo) y DB
make k8s-down-minio      # sólo minio-dev
make k8s-down-argo       # sólo argo
make k8s-down-db         # sólo DB
make port-forward        # abre PF (MinIO/Argo/DB)
make port-forward-stop   # cierra PF
make port-forward-status # estado PF
make backend             # sólo backend (8000)
make frontend            # sólo frontend (3000)
make workflow            # envía workflow de ejemplo a Argo
```

---

### Puertos y URLs
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9090`
- Argo UI: `https://localhost:2746`
- Postgres: `localhost:5432`
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

---

### Variables útiles
- `MINIKUBE_PROFILE` — perfil de minikube (por defecto `minikube`)
- `K8S_CONTEXT` — contexto de kubectl (por defecto `minikube`)
- `DATABASE_URL` — conexión a Postgres (ej. `postgresql+psycopg://syntheticdata:syntheticdata@localhost:5432/syntheticdata`)
- `RUN_MIGRATIONS` — crea tablas del registry al iniciar (`true`/`false`)

Ejemplo:
```bash
MINIKUBE_PROFILE=devbox K8S_CONTEXT=minikube \
  ./scripts/dev/k8s-up.sh
```

### Backend, MinIO y Argo CLI

El backend necesita credenciales de MinIO para poder aceptar las subidas de archivos desde el frontend y para guardar el manifiesto del workflow antes de enviarlo a Argo. Las definiciones de workflows ahora se guardan en PostgreSQL. Las variables más importantes relacionadas con MinIO son:

- `MINIO_ENDPOINT` — endpoint del servicio S3 (por defecto `localhost:9000`)
- `MINIO_ACCESS_KEY` — access key obligatoria
- `MINIO_SECRET_KEY` — secret key obligatoria
- `MINIO_SECURE` — `false` por defecto (útil para HTTPS en entornos reales)
- `MINIO_REGION` — opcional, sólo si tu MinIO/S3 la requiere
- `MINIO_INPUT_BUCKET` — bucket por defecto donde se guardan los ficheros subidos (`argo-artifacts` si no se configura)

El backend crea automáticamente `MINIO_INPUT_BUCKET` si no existe. En local, cuando usas los manifests incluidos, las credenciales por defecto son `minioadmin`/`minioadmin`:

```bash
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
make backend
```

Los ficheros que se suben desde el frontend se organizan dentro del bucket mediante la convención:

```
sessions/<sessionId>/nodes/<nodeId>/<nombre>-<uuid>.<ext>
```

El `sessionId` se genera en el navegador en cada carga del editor y `nodeId` corresponde al identificador del nodo (o su `templateName` cuando aplica). De esta forma todos los artefactos quedan agrupados por sesión y nodo sin necesidad de configurar buckets adicionales.

Además, el backend ejecuta directamente `argo submit` usando el manifiesto generado. Para ello necesita acceso al binario `argo` y permite configurarlo mediante estas variables:

- `ARGO_CLI_PATH` — ruta al binario (`argo` por defecto)
- `ARGO_NAMESPACE` — namespace de Kubernetes donde se envían los workflows (`argo` por defecto)
- `ARGO_SUBMIT_EXTRA_ARGS` — cadena opcional con flags extra (por ejemplo `--serviceaccount custom-sa`)

La respuesta del endpoint incluye el nombre del workflow lanzado, el namespace y la ruta dentro de MinIO donde se guardó el YAML (`sessions/<sessionId>/workflow/<filename>`), para poder consultarlo más adelante si es necesario.

### Enviar el DAG generado en el canvas

El editor (`frontend`) envía el flujo directamente a Argo usando el botón **Enviar Workflow**. El proceso es:

1. Construye el flujo en el canvas conectando nodos del catálogo.
2. Sube los artefactos necesarios a través de los nodos de entrada.
3. Pulsa **Enviar Workflow** (parte superior derecha). El frontend envía el grafo actual y el `sessionId` al backend.
4. El backend genera el Workflow de Argo, guarda el YAML en MinIO bajo `sessions/<sessionId>/workflow/<filename>` y ejecuta `argo submit` en el namespace configurado.
5. La interfaz muestra el nombre del workflow creado y la ubicación del manifiesto en MinIO para referencia rápida.

Además, el editor dispone ahora de una terminal inferior plegable que se abre automáticamente tras cada envío y durante la ejecución. Los pods continúan archivando sus logs en MinIO, pero la interfaz los obtiene en tiempo (casi) real consultando la API de Argo (`GET /workflow/logs/stream`), que recibe el `workflowName`, un cursor Base64 incremental y parámetros opcionales como `namespace`, `tailLines` o `container`.

> **Nota:** El backend genera plantillas embebidas para cada nodo del catálogo, así que no necesitas WorkflowTemplates registradas en el clúster.

#### Seguimiento de estado con la API de Argo Server

El backend también consulta el estado de cada workflow usando directamente el API REST expuesto por `argo-server`. De esta forma, el canvas puede colorear cada nodo según su fase actual (`Pendiente`, `En ejecución`, `Completado`, `Error`, etc.) mientras el workflow avanza.

Variables de entorno asociadas:

- `ARGO_SERVER_BASE_URL` — URL base del servidor (por defecto `https://localhost:2746` si haces port-forward del `argo-server`)
- `ARGO_SERVER_AUTH_TOKEN` — token Bearer opcional cuando el servidor exige autenticación
- `ARGO_SERVER_INSECURE_SKIP_VERIFY` — ajústalo a `true` para omitir la verificación TLS en entornos dev/self-signed (se omite automáticamente cuando la URL apunta a `localhost`)
- `ARGO_SERVER_TIMEOUT_SECONDS` — timeout en segundos para las peticiones al API (por defecto `10`)

El frontend realiza polling periódico contra `GET /workflow/status`. Si el servidor todavía no registra el workflow y responde `404`, el backend lo traduce a un `null` y la UI reintenta automáticamente hasta que Argo lo cree. Una vez finalizada la ejecución, se conserva el último estado por nodo para revisión.

### Nodos de salida y previsualización de artefactos

- El catálogo incluye un nodo `data-output` de tipo salida. Arrástralo al lienzo y conéctalo a la salida de cualquier nodo que produzca artefactos.
- El inspector muestra la ruta esperada en MinIO (`bucket` y `key`), así como metadatos básicos (tamaño, content-type, nodo de origen).
- Desde el inspector puedes descargar el fichero directamente (`GET /artifacts/download`) o solicitar una previsualización de hasta 64 KB (`GET /artifacts/preview`). Los ficheros de texto (`.csv`, `.json`, etc.) se renderizan inline; si el contenido es mayor, se indica que está truncado.
- La aplicación consulta automáticamente las ubicaciones con `POST /workflow/output-artifacts`, reutilizando el mismo grafo que se envía a Argo para asegurar que la ruta generada es exactamente la misma que usará el workflow.


---

### Notas y troubleshooting
- Si Argo pide login, espera a que el `k8s-up.sh` haya aplicado el parche `--auth-mode=server` y que el rollout termine.
- Si un port-forward no arranca, revisa logs en `.tmp/dev/*.log`.
- Si cambias los manifests en `deploy/`, vuelve a correr `make k8s-up` o aplica los YAML con `kubectl apply -f ...`.


