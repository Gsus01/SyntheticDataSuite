### Requisitos Previos
* Tener instalado y corriendo `minikube`. Puedes iniciarlo con `minikube start`.
* Tener instalado `kubectl`.

### 1. Despliegue de MinIO
MinIO se usará como nuestro repositorio de artefactos, es decir, el lugar donde Argo Workflows guardará las salidas y resultados de los flujos de trabajo.

#### 1.1. Creación del Namespace y Pod de MinIO
El fichero `minio-dev.yaml` se encarga de dos cosas:
* **Namespace**: Crea un `Namespace` llamado `minio-dev`. Los namespaces en Kubernetes son una forma de organizar y aislar recursos.
* **Pod**: Despliega un `Pod` de MinIO en el namespace `minio-dev`. Un Pod es la unidad de despliegue más pequeña en Kubernetes y contiene uno o más contenedores. Este Pod utiliza una imagen de MinIO y monta un volumen local de la máquina anfitriona (`/home/gsus/minio/`) para almacenar los datos de forma persistente.

Para aplicar este fichero, ejecuta:
```bash
kubectl apply -f deploy/minio/minio-dev.yaml
```

#### 1.2. Creación del Servicio de MinIO
El fichero `minio-service.yaml` crea un `Service` de Kubernetes. Un servicio expone una forma de acceder a los Pods, en este caso, al Pod de MinIO. Este servicio (`minio`) permitirá que otros componentes dentro del clúster (como Argo) se comuniquen con MinIO a través de un nombre de DNS estable (`minio.minio-dev.svc`).

Aplica el fichero con:
```bash
kubectl apply -f deploy/minio/minio-service.yaml
```

### 2. Despliegue de Argo Workflows
Ahora instalaremos Argo Workflows, el motor que orquestará nuestros flujos de trabajo.

#### 2.1. Creación del Namespace e Instalación
Primero, creamos un `Namespace` para Argo y luego aplicamos la configuración de instalación oficial.
```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
```

#### 2.2. Configuración de Roles y Permisos
Para que los flujos de trabajo (workflows) puedan crear y actualizar sus propios resultados, necesitamos darles permisos específicos.
* `argo-workflow-role.yaml`: Define un `Role` llamado `workflow-role` en el namespace `argo`. Este rol concede permisos para crear y parchear (`create`, `patch`) recursos de tipo `workflowtaskresults` dentro del grupo de API `argoproj.io`.
* `argo-workflow-rolebinding.yaml`: Crea un `RoleBinding` que asigna el `workflow-role` a la `ServiceAccount` por defecto (`default`) del namespace `argo`. Esto significa que cualquier Pod que se ejecute con esta ServiceAccount (que es lo que harán nuestros workflows por defecto) tendrá los permisos definidos en el rol.

Aplica ambos ficheros:
```bash
kubectl apply -f deploy/argo/argo-workflow-role.yaml
kubectl apply -f deploy/argo/argo-workflow-rolebinding.yaml
```

#### 2.3. Ajuste del Modo de Autenticación para el desarrollo en local

Por defecto, la interfaz de Argo Workflows requiere un token de autenticación para iniciar sesión. Para un entorno de desarrollo local como este, podemos cambiar el modo de autenticación a server para desactivar esta pantalla de login y acceder directamente.

Aplica este parche para modificar la configuración del servidor de Argo:

```bash
kubectl -n argo patch deployment argo-server --type=json -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--auth-mode=server"}]'
```
Este comando hará que el Pod de argo-server se reinicie automáticamente con la nueva configuración. Puedes esperar a que el proceso termine con el siguiente comando:

```bash
kubectl -n argo rollout status deployment/argo-server
```

Luego tienes que conseguir el token de acceso:
```bash
kubectl -n argo create token argo-server
```

### 3. Conexión de Argo con MinIO
Finalmente, configuramos Argo para que sepa dónde encontrar y cómo autenticarse con nuestro servidor MinIO.

#### 3.1. Creación de las Credenciales
El fichero `minio-creds-secret.yaml` crea un `Secret` de Kubernetes llamado `minio-creds` en el namespace `argo`. Este secreto almacena de forma segura las credenciales de acceso a MinIO (`accessKey` y `secretKey`).

Aplica el fichero:
```bash
kubectl apply -f deploy/argo/minio-creds-secret.yaml
```

#### 3.2. Configuración del Repositorio de Artefactos
Argo instala por defecto un repositorio de artefactos llamado `my-minio-cred` que apunta a `minio:9000` y usa las claves `accesskey/secretkey`. Si no lo actualizas, los workflows seguirán buscando los ficheros en ese bucket inexistente aunque hayas cargado los datos en `argo-artifacts`.

Para que todo apunte a tu despliegue de MinIO debemos actualizar **dos** `ConfigMaps`:

1. `artifact-repositories-configmap.yaml` sobreescribe el `ConfigMap` global `artifact-repositories` (usado para inputs/outputs vía `argo-server`):
   ```bash
   kubectl apply -f deploy/argo/artifact-repositories-configmap.yaml
   ```
2. `artifact-repository-configmap.yaml` mantiene en sincronía la configuración del `workflow-controller`:
   ```bash
   kubectl apply -f deploy/argo/artifact-repository-configmap.yaml
   ```

Ambos ficheros configuran:
* **endpoint**: `minio.minio-dev.svc:9000`
* **bucket**: `argo-artifacts`
* **insecure**: `true`
* **accessKeySecret** / **secretKeySecret**: usan el secreto `minio-creds` con claves `accessKey` y `secretKey` (respeta mayúsculas/minúsculas).

> Si el controlador o el servidor de Argo ya estaban corriendo, reinícialos para que recojan los nuevos valores:
> ```bash
> kubectl -n argo rollout restart deployment/workflow-controller
> kubectl -n argo rollout restart deployment/argo-server
> ```

### 4. Lanzamiento y Acceso
Con todo configurado, ya puedes empezar a lanzar workflows.

* **Acceder a la Interfaz de Argo:**
    ```bash
    kubectl -n argo port-forward deployment/argo-server 2746:2746
    ```
    Ahora puedes abrir `https://localhost:2746` en tu navegador.

* **Acceder a la Consola de MinIO:**
    ```bash
    kubectl -n minio-dev port-forward pod/minio 9090:9090 9000:9000
    ```
    Puedes acceder a la consola en `http://localhost:9090`. Los artefactos de tus workflows aparecerán en el bucket `argo-artifacts`.

### 5. Ejecutar el Workflow de Prueba HMM

Una vez que tengas todo configurado, puedes probar el workflow de generación de datos sintéticos con HMM:

```bash
# Ejecutar el workflow simplificado (solo HMM)
argo submit deploy/general_workflow.yaml -n argo

# Ejecutar con logs en tiempo real
argo submit deploy/general_workflow.yaml --log -n argo

# Ver el estado de los workflows
argo list

# Ver detalles del workflow
argo get <workflow-name>
```

**Nota**: Este workflow utiliza archivos de ejemplo incluidos en las imágenes Docker. Para usar tus propios datos, puedes subirlos a MinIO usando la consola web (http://localhost:9090) con credenciales `minioadmin:minioadmin`.

### 6. Troubleshooting Común

#### Verificar que MinIO esté funcionando:
```bash
kubectl -n minio-dev get pods
kubectl -n minio-dev logs minio
```

#### Verificar que Argo esté funcionando:
```bash
kubectl -n argo get pods
kubectl -n argo logs deployment/argo-server
```

#### Si el workflow falla:
```bash
# Ver logs detallados del workflow
argo logs <workflow-name>

# Ver logs de un paso específico
argo logs <workflow-name> -c preprocessing
argo logs <workflow-name> -c train-hmm-model
```

---

### Atajo para desarrollo
Para evitar pasos manuales en local, usa los scripts de `scripts/dev/` o los objetivos del `Makefile` descritos en `docs/dev-workflow.md`. Por ejemplo:

```bash
make dev            # Levanta MinIO + Argo + port-forwards + backend + frontend
make k8s-up-argo    # Sólo Argo
make k8s-up-minio   # Sólo MinIO
```
