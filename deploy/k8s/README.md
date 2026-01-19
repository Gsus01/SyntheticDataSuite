# Kubernetes Deployment

This directory contains Kubernetes manifests to deploy the SyntheticDataSuite backend and frontend.

## Prerequisites

1. **minikube** running with Argo Workflows, MinIO, and PostgreSQL already deployed
2. **kubectl** configured to access the cluster
3. Docker images built in minikube's Docker daemon

## Quick Start (Recommended: With Ingress)

### 0. Deploy dependencies (MinIO, Argo, Postgres)

```bash
./scripts/dev/k8s-up.sh
```

The recommended setup uses an **Ingress** to serve both frontend and backend under the same domain. This avoids CORS issues and simplifies the architecture.

### 1. Enable Ingress addon

```bash
minikube addons enable ingress
```

Wait for the ingress controller to be ready:

```bash
kubectl get pods -n ingress-nginx
# Wait until the controller pod is Running
```

### 2. Build the images

```bash
# Configure Docker to use minikube's daemon
eval $(minikube docker-env)

# Build backend and frontend (uses /api as default backend URL)
SKIP_COMPONENTS=1 ./build_all.sh
```

### 3. Deploy to Kubernetes

```bash
# Apply all manifests (run twice if namespace error on first run)
kubectl apply -f deploy/k8s/
kubectl apply -f deploy/k8s/  # Second run to ensure all resources are created
```

### 4. Configure local DNS

Add the minikube IP to your hosts file:

```bash
echo "$(minikube ip) syntheticdata.local" | sudo tee -a /etc/hosts
```

### 5. Access the application

Open http://syntheticdata.local in your browser.

- Frontend: `http://syntheticdata.local/`
- Backend API: `http://syntheticdata.local/api/health`

---

## Alternative: Without Ingress (Port Forwarding)

If you prefer not to use Ingress, you can use port-forwarding:

### 1. Build with localhost URL

```bash
eval $(minikube docker-env)
BACKEND_URL=http://localhost:8000 SKIP_COMPONENTS=1 ./build_all.sh
```

### 2. Deploy (skip ingress.yaml)

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/backend-rbac.yaml
kubectl apply -f deploy/k8s/backend-secrets.yaml
kubectl apply -f deploy/k8s/backend-configmap.yaml
kubectl apply -f deploy/k8s/backend-deployment.yaml
kubectl apply -f deploy/k8s/backend-service.yaml
kubectl apply -f deploy/k8s/frontend-deployment.yaml
kubectl apply -f deploy/k8s/frontend-service.yaml
```

### 3. Port forward both services

```bash
# Terminal 1: Backend
kubectl port-forward -n syntheticdata svc/syntheticdata-backend 8000:8000

# Terminal 2: Frontend
kubectl port-forward -n syntheticdata svc/syntheticdata-frontend 3000:3000
```

### 4. Access

Open http://localhost:3000

---

## Architecture

### With Ingress (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser                                  │
│                            │                                     │
│              http://syntheticdata.local                         │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ingress Controller                            │
│                    (nginx in minikube)                           │
│                            │                                     │
│         ┌──────────────────┴──────────────────┐                 │
│         │                                     │                 │
│    /api/*                                   /*                  │
│    (rewrite to /*)                                              │
│         │                                     │                 │
│         ▼                                     ▼                 │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │     Backend     │              │    Frontend     │          │
│  │  (FastAPI:8000) │              │  (Next.js:3000) │          │
│  └─────────────────┘              └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Without Ingress

```
┌─────────────────┐    ┌─────────────────┐
│     Browser     │    │     Browser     │
│   :3000 (UI)    │    │   :8000 (API)   │
└────────┬────────┘    └────────┬────────┘
         │                      │
    port-forward           port-forward
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend     │
│  (Next.js:3000) │───▶│  (FastAPI:8000) │
└─────────────────┘    └─────────────────┘
```

---

## Configuration

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Postgres connection string | `postgresql+psycopg://syntheticdata:syntheticdata@postgres.syntheticdata.svc.cluster.local:5432/syntheticdata` |
| `RUN_MIGRATIONS` | Create registry tables on startup | `true` |
| `MINIO_ENDPOINT` | MinIO server address | `minio.minio-dev.svc.cluster.local:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | (from secret) |
| `MINIO_SECRET_KEY` | MinIO secret key | (from secret) |
| `MINIO_SECURE` | Use HTTPS for MinIO | `false` |
| `ARGO_SERVER_BASE_URL` | Argo Server REST API URL | `https://argo-server.argo.svc.cluster.local:2746` |
| `ARGO_NAMESPACE` | Namespace for Argo workflows | `argo` |
| `SKIP_IMAGE_VALIDATION` | Skip Docker image validation | `true` |

### Frontend Build Arguments

| Argument | Description | Recommended |
|----------|-------------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | Backend API URL | `/api` (for Ingress) |

---

## File Structure

```
deploy/k8s/
├── namespace.yaml           # syntheticdata namespace
├── ingress.yaml             # Ingress routing (frontend + backend)
├── backend-configmap.yaml   # Backend environment config
├── backend-secrets.yaml     # Backend credentials (edit before use)
├── backend-rbac.yaml        # ServiceAccount & permissions
├── backend-deployment.yaml  # Backend pod spec
├── backend-service.yaml     # Backend ClusterIP service
├── frontend-deployment.yaml # Frontend pod spec
├── frontend-service.yaml    # Frontend ClusterIP service
└── README.md                # This file
```

---

## Troubleshooting

### Check pod status

```bash
kubectl get pods -n syntheticdata
kubectl describe pod -n syntheticdata <pod-name>
kubectl logs -n syntheticdata <pod-name>
```

### Ingress not working

1. Check if ingress addon is enabled:
   ```bash
   minikube addons list | grep ingress
   ```

2. Check ingress controller is running:
   ```bash
   kubectl get pods -n ingress-nginx
   ```

3. Check ingress resource:
   ```bash
   kubectl get ingress -n syntheticdata
   kubectl describe ingress syntheticdata -n syntheticdata
   ```

4. Verify /etc/hosts has the correct IP:
   ```bash
   cat /etc/hosts | grep syntheticdata
   minikube ip  # Should match the IP in hosts file
   ```

### Backend can't connect to MinIO

Verify MinIO is running and the endpoint is correct:

```bash
kubectl get svc -n minio-dev
# Update MINIO_ENDPOINT in backend-configmap.yaml if needed
```

### Backend can't submit Argo workflows

Check RBAC permissions:

```bash
kubectl auth can-i create workflows.argoproj.io -n argo \
  --as=system:serviceaccount:syntheticdata:syntheticdata-backend
```

### 502 Bad Gateway

The backend might not be ready yet. Check:

```bash
kubectl get pods -n syntheticdata
kubectl logs -n syntheticdata -l app.kubernetes.io/name=backend
```

---

## Development Workflow

### Option 1: Full Kubernetes (recommended for integration testing)

```bash
# Build and deploy
eval $(minikube docker-env)
SKIP_COMPONENTS=1 ./build_all.sh
kubectl rollout restart deployment -n syntheticdata

# Access via Ingress
open http://syntheticdata.local
```

### Option 2: Local development (faster iteration)

Run backend and frontend outside containers, but use Kubernetes services:

```bash
# Backend (in backend/ directory)
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (in frontend/ directory)
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

### Option 3: Hybrid

Run frontend locally, backend in Kubernetes:

```bash
# Port-forward backend
kubectl port-forward -n syntheticdata svc/syntheticdata-backend 8000:8000

# Run frontend locally pointing to backend
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

---

## Deploying to a New Machine

1. **Install prerequisites**:
   - minikube
   - kubectl
   - Docker

2. **Start minikube**:
   ```bash
   minikube start
   ```

3. **Deploy Argo Workflows** (if not already):
   ```bash
   kubectl create namespace argo
   kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/install.yaml
   ```

4. **Deploy MinIO** (if not already):
   ```bash
   kubectl apply -f deploy/minio/
   ```

5. **Enable Ingress**:
   ```bash
   minikube addons enable ingress
   ```

6. **Build images**:
   ```bash
   eval $(minikube docker-env)
   ./build_all.sh  # Or SKIP_COMPONENTS=1 for just backend/frontend
   ```

7. **Deploy application**:
   ```bash
   kubectl apply -f deploy/k8s/
   kubectl apply -f deploy/k8s/  # Run twice for namespace timing
   ```

8. **Configure DNS**:
   ```bash
   echo "$(minikube ip) syntheticdata.local" | sudo tee -a /etc/hosts
   ```

9. **Verify**:
   ```bash
   kubectl get pods -n syntheticdata
   curl http://syntheticdata.local/api/health
   ```

10. **Open browser**: http://syntheticdata.local
