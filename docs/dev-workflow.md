### Guía de Desarrollo: levantar y trabajar con la suite

Este documento resume cómo arrancar el entorno de desarrollo local de forma rápida usando los scripts y el Makefile. La idea es minimizar pasos manuales y permitir arrancar sólo las piezas que necesites (MinIO, Argo, backend, frontend).

#### Requisitos
- minikube y kubectl instalados
- Node 18+ y npm
- Python 3.10+ (para backend) y `uvicorn`

#### Comando rápido (todo)
```bash
make dev
```
Arranca: MinIO + Argo en minikube, abre los puertos (9090/9000 para MinIO, 2746 para Argo) y lanza frontend (3000) y backend (8000).

---

### Scripts disponibles

Todos los scripts están en `scripts/dev/`.

1) `k8s-up.sh` — despliegue en Kubernetes (minikube)
- Uso básico: `./scripts/dev/k8s-up.sh`
- Sólo un componente:
  - `./scripts/dev/k8s-up.sh --only minio`
  - `./scripts/dev/k8s-up.sh --only argo`
- Comportamiento:
  - Arranca minikube si no está corriendo (perfil configurable con `MINIKUBE_PROFILE`)
  - Aplica manifests de MinIO y/o Argo
  - Ajusta Argo para `--auth-mode=server` (modo dev)
  - Espera a que los deployments estén listos
  - Si `kubectl` no existe en el PATH, usa automáticamente `minikube kubectl --`

2) `k8s-down.sh` — limpieza de namespaces
- Uso básico: `./scripts/dev/k8s-down.sh`
- Sólo un componente:
  - `./scripts/dev/k8s-down.sh --only minio`
  - `./scripts/dev/k8s-down.sh --only argo`

3) `port-forward.sh` — gestionar port-forwards y estados
- Estado: `./scripts/dev/port-forward.sh status`
- Iniciar:
  - `./scripts/dev/port-forward.sh start`
  - Selectivo: `./scripts/dev/port-forward.sh start --only argo`
- Parar:
  - `./scripts/dev/port-forward.sh stop`
  - Selectivo: `./scripts/dev/port-forward.sh stop --only minio`
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

---

### Makefile (atajos)

```bash
make dev                 # orquesta todo
make k8s-up              # sube MinIO + Argo
make k8s-up-minio        # sólo MinIO
make k8s-up-argo         # sólo Argo
make k8s-down            # elimina ambos namespaces
make k8s-down-minio      # sólo minio-dev
make k8s-down-argo       # sólo argo
make port-forward        # abre PF (MinIO/Argo)
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
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

---

### Variables útiles
- `MINIKUBE_PROFILE` — perfil de minikube (por defecto `minikube`)
- `K8S_CONTEXT` — contexto de kubectl (por defecto `minikube`)

Ejemplo:
```bash
MINIKUBE_PROFILE=devbox K8S_CONTEXT=minikube \
  ./scripts/dev/k8s-up.sh
```

---

### Notas y troubleshooting
- Si Argo pide login, espera a que el `k8s-up.sh` haya aplicado el parche `--auth-mode=server` y que el rollout termine.
- Si un port-forward no arranca, revisa logs en `.tmp/dev/*.log`.
- Si cambias los manifests en `deploy/`, vuelve a correr `make k8s-up` o aplica los YAML con `kubectl apply -f ...`.


