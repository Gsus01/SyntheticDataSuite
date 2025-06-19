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

### 3. Conexión de Argo con MinIO
Finalmente, configuramos Argo para que sepa dónde encontrar y cómo autenticarse con nuestro servidor MinIO.

#### 3.1. Creación de las Credenciales
El fichero `minio-creds-secret.yaml` crea un `Secret` de Kubernetes llamado `minio-creds` en el namespace `argo`. Este secreto almacena de forma segura las credenciales de acceso a MinIO (`accessKey` y `secretKey`).

Aplica el fichero:
```bash
kubectl apply -f deploy/argo/minio-creds-secret.yaml
```

#### 3.2. Configuración del Repositorio de Artefactos
El fichero `artifact-repository-configmap.yaml` modifica el `ConfigMap` principal del controlador de Argo (`workflow-controller-configmap`). Dentro de este `ConfigMap`, se especifica la configuración del repositorio de artefactos por defecto:
* **endpoint**: Apunta al servicio de MinIO que creamos anteriormente (`minio.minio-dev.svc:9000`).
* **bucket**: Indica el nombre del bucket que se usará (`argo-artifacts`). Argo lo creará si no existe.
* **insecure**: Se establece en `true` para permitir conexiones sin SSL, lo cual es común en entornos de desarrollo local como Minikube.
* **accessKeySecret** y **secretKeySecret**: Referencian el `Secret` `minio-creds` para obtener las credenciales de autenticación.

Aplica el fichero:
```bash
kubectl apply -f deploy/argo/artifact-repository-configmap.yaml
```

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