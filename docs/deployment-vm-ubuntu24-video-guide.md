# Guia de despliegue en una VM Ubuntu 24.04

Esta guia esta pensada para una grabacion de pantalla sin voz. Sigue los bloques en orden y deja que cada comando termine antes de pasar al siguiente.

Resultado esperado:

- Frontend en `http://syntheticdata.local`
- Backend en `http://syntheticdata.local/api/health`
- MinIO en `http://localhost:9090`
- Argo en `https://localhost:2746`

Esta guia valida que la plataforma queda desplegada y accesible. No incluye la ejecucion end-to-end de `deploy/general_workflow.yaml`.

## 0. Orden sugerido para la grabacion

Si quieres que el video sea facil de seguir, usa este orden:

1. Mostrar version de Ubuntu, arquitectura y memoria.
2. Instalar Docker, kubectl y minikube.
3. Clonar el repositorio.
4. Arrancar minikube y ejecutar `make k8s-deploy`.
5. Configurar `syntheticdata.local`.
6. Abrir frontend, MinIO y Argo.
7. Cerrar con los comandos de apagado.

Comandos utiles para la apertura del video:

```bash
lsb_release -d
uname -m
free -h
```

## 1. Requisitos de la VM

Minimo recomendado para la VM:

- Ubuntu 24.04 LTS
- 4 vCPU
- 8 GB RAM
- 40 GB de disco
- Acceso a internet
- Usuario con `sudo`
- Navegador dentro de la VM para comprobar la interfaz final

Nota:

- Esta guia asume `amd64` o `arm64`.
- Si trabajas por SSH sin escritorio, el flujo principal cambia. Para este video es mas simple usar el navegador dentro de la propia VM.

## 2. Instalar paquetes base

```bash
sudo apt update
sudo apt install -y git make curl ca-certificates gnupg conntrack
```

## 3. Instalar Docker Engine

Comandos basados en la guia oficial de Docker para Ubuntu:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Agregar tu usuario al grupo `docker` para no depender de `sudo`:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Verificacion rapida:

```bash
docker run hello-world
docker version
```

## 4. Instalar kubectl y minikube

Detectar arquitectura de la VM:

```bash
ARCH="$(dpkg --print-architecture)"
case "$ARCH" in
  amd64)
    K8S_ARCH=amd64
    MINIKUBE_DEB=minikube_latest_amd64.deb
    ;;
  arm64)
    K8S_ARCH=arm64
    MINIKUBE_DEB=minikube_latest_arm64.deb
    ;;
  *)
    echo "Arquitectura no soportada por esta guia: $ARCH"
    exit 1
    ;;
esac
echo "$ARCH"
```

Instalar `kubectl`:

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/${K8S_ARCH}/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
kubectl version --client
```

Instalar `minikube`:

```bash
curl -LO "https://storage.googleapis.com/minikube/releases/latest/${MINIKUBE_DEB}"
sudo dpkg -i "${MINIKUBE_DEB}"
rm "${MINIKUBE_DEB}"
minikube version
```

## 5. Clonar el repositorio

```bash
git clone https://github.com/Gsus01/SyntheticDataSuite.git
cd SyntheticDataSuite
```

Opcional para mostrar en el video:

```bash
git rev-parse --short HEAD
ls
```

## 6. Arrancar minikube con recursos suficientes

Aunque `make k8s-deploy` ya puede arrancar minikube, para una VM nueva conviene dejar el cluster listo antes de desplegar.

```bash
minikube config set driver docker
minikube start --driver=docker --cpus=4 --memory=8192
kubectl get nodes
```

## 7. Desplegar SyntheticDataSuite

Este comando hace todo el flujo principal:

- habilita Ingress si hace falta,
- despliega MinIO, Argo y PostgreSQL,
- construye las imagenes de backend y frontend dentro de minikube,
- aplica los manifests de la aplicacion,
- abre port-forwards para MinIO y Argo.

```bash
make k8s-deploy
```

Tiempo normal de espera:

- primera ejecucion: varios minutos,
- depende de la red y de la descarga de imagenes.

## 8. Configurar `syntheticdata.local`

Obtener la IP del cluster:

```bash
minikube ip
```

Agregar la entrada en `/etc/hosts`:

```bash
echo "$(minikube ip) syntheticdata.local" | sudo tee -a /etc/hosts
```

Si ya existia una entrada vieja, edita `/etc/hosts` y deja solo una linea correcta para `syntheticdata.local`.

## 9. Verificacion final

### 9.1. Estado de pods

```bash
kubectl get pods -n minio-dev
kubectl get pods -n argo
kubectl get pods -n syntheticdata
```

Debes ver los pods principales en estado `Running` o `Completed` segun corresponda.

### 9.2. Backend saludable

```bash
curl http://syntheticdata.local/api/health
```

### 9.3. Abrir las interfaces

Abre estas URLs en el navegador de la VM:

- `http://syntheticdata.local`
- `http://syntheticdata.local/api/health`
- `http://localhost:9090`
- `https://localhost:2746`

Que deberias comprobar:

- `syntheticdata.local` carga el frontend.
- `/api/health` responde desde FastAPI.
- MinIO abre en `localhost:9090`.
- Argo abre en `localhost:2746`.

Credenciales esperadas de MinIO en local:

- usuario: `minioadmin`
- password: `minioadmin`

## 10. Comandos utiles para el video

Ver recursos desplegados:

```bash
kubectl get all -n syntheticdata
kubectl get ingress -n syntheticdata
kubectl get svc -n syntheticdata
```

Ver logs si algo tarda demasiado:

```bash
kubectl logs -n syntheticdata -l app.kubernetes.io/name=backend
kubectl logs -n syntheticdata -l app.kubernetes.io/name=frontend
kubectl logs -n argo deployment/argo-server
```

## 11. Apagar el entorno

Detener solo la aplicacion:

```bash
make k8s-stop
```

Detener tambien MinIO, Argo y Postgres:

```bash
make k8s-down
```

Parar minikube por completo:

```bash
minikube stop
```

## 12. Troubleshooting rapido

### Docker no funciona sin sudo

Ejecuta:

```bash
newgrp docker
docker run hello-world
```

Si sigue fallando, cierra sesion de la VM y vuelve a entrar.

### `make k8s-deploy` tarda mucho

Es normal en la primera ejecucion porque descarga imagenes y construye backend/frontend.

### `syntheticdata.local` no abre

Revisa:

```bash
minikube ip
grep syntheticdata.local /etc/hosts
kubectl get ingress -n syntheticdata
kubectl get pods -n ingress-nginx
```

La IP en `/etc/hosts` debe coincidir con `minikube ip`.

### Backend o frontend no estan listos

Revisa:

```bash
kubectl get pods -n syntheticdata
kubectl describe pod -n syntheticdata <nombre-del-pod>
kubectl logs -n syntheticdata <nombre-del-pod>
```

### MinIO o Argo no abren en localhost

Revisa si los port-forwards siguen activos:

```bash
./scripts/dev/port-forward.sh status
```

Si hace falta, vuelvelos a abrir:

```bash
make port-forward
```

### Acceso remoto por SSH

Si no tienes navegador dentro de la VM, esta guia deja de ser el camino mas simple. Para un video corto, usa un escritorio en la VM. Si aun asi quieres operar en remoto, tendras que resolver por separado el acceso a `syntheticdata.local` y los tuneles hacia `localhost:9090` y `localhost:2746`.

## 13. Nota sobre workflows de ejemplo

Esta guia termina cuando la plataforma queda desplegada y accesible. Si despues quieres validar `deploy/general_workflow.yaml`, ten en cuenta que ese workflow referencia imagenes de componentes con tag `:v2`, mientras que `make k8s-deploy` solo construye backend y frontend.

## 14. Referencias oficiales

Las instrucciones de instalacion usadas en esta guia se han alineado con estas fuentes oficiales consultadas el 2026-03-11:

- Docker Ubuntu install: https://docs.docker.com/engine/install/ubuntu/
- Docker Linux post-install: https://docs.docker.com/engine/install/linux-postinstall/
- kubectl on Linux: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
- minikube get started: https://minikube.sigs.k8s.io/docs/start/
- minikube Docker driver: https://minikube.sigs.k8s.io/docs/drivers/docker/
