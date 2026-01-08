# 8. Seguridad

## Introducción

La seguridad en una plataforma de orquestación de workflows abarca múltiples capas: desde las imágenes de contenedor hasta la autenticación de usuarios. Este apartado describe las medidas de seguridad adoptadas en la arquitectura objetivo, con decisiones concretas de herramientas coherentes con el resto del stack tecnológico.

El prototipo actual no implementa la mayoría de estas medidas, ya que su objetivo es validar la viabilidad técnica en un entorno controlado. Sin embargo, cualquier despliegue en producción debe incorporar los controles aquí descritos.

---

## 8.1. Seguridad en Contenedores

### Decisión: Trivy como Escáner de Vulnerabilidades

Se adopta **Trivy** (Aqua Security) como herramienta de escaneo de vulnerabilidades en imágenes. La elección se basa en:

- **Gratuito y open source**: sin costes de licencia, comunidad activa.
- **Integración sencilla**: una línea en el pipeline CI (`trivy image <imagen>`).
- **Base de datos CVE actualizada**: Trivy mantiene su propia base de datos sincronizada.
- **Cobertura completa**: escanea vulnerabilidades del SO, dependencias de lenguaje y misconfigurations de Dockerfile.

**Alternativas descartadas**:
- *Grype*: similar en funcionalidad pero menos documentación y comunidad.
- *Snyk*: excelente producto pero modelo freemium con limitaciones en tier gratuito.
- *Clair*: arquitectura más compleja (requiere desplegar un servicio separado).

### Política de Bloqueo

El pipeline bloqueará el push de imágenes si Trivy detecta:
- Vulnerabilidades **críticas** (CVSS ≥ 9.0): bloqueo inmediato.
- Vulnerabilidades **altas** sin parche disponible: bloqueo.
- Vulnerabilidades **altas** con parche: advertencia, permitido con excepción documentada.

### Buenas Prácticas de Construcción

Las imágenes de la plataforma deben seguir estas normas:
- Usar imágenes base mínimas (`python:3.11-slim`, `node:20-alpine`, o `distroless` cuando sea viable).
- Ejecutar el proceso con usuario no-root (directiva `USER` en Dockerfile).
- No incluir secretos ni credenciales en la imagen.
- Pinear versiones de dependencias y usar digests para imágenes base.

### Estado del Prototipo

No implementado. Las imágenes se construyen sin escaneo de seguridad.

---

## 8.2. Seguridad en Kubernetes

### RBAC: Mínimo Privilegio por Service Account

Cada componente de la plataforma tiene su propio Service Account con permisos estrictamente necesarios:

**backend-sa**:
- `get`, `list`, `create`, `delete` sobre `workflows.argoproj.io` en su namespace.
- `get` sobre `secrets` y `configmaps` de configuración.

**argo-workflow-controller-sa** (gestionado por Argo):
- Permisos para crear pods efímeros, gestionar artifacts.

**frontend-sa**:
- Sin permisos especiales (el frontend no interactúa directamente con la API de Kubernetes).

Los entornos (dev, pre, prod) están aislados en namespaces separados. Ninguna aplicación usa `cluster-admin`.

### Decisión: Network Policies con Política Deny-All por Defecto

Se aplica una política **deny-all** por defecto en cada namespace, permitiendo solo el tráfico explícitamente declarado:

```yaml
# Deny all ingress by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
spec:
  podSelector: {}
  policyTypes:
    - Ingress
```

Sobre esta base, se añaden políticas específicas:

| Destino | Orígenes Permitidos |
|:--------|:--------------------|
| Backend | Ingress Controller, Frontend |
| MinIO | Backend, Workflow Pods |
| Argo Server | Backend |
| Workflow Pods | Solo egress a MinIO |

### Pod Security Standards: Nivel Restricted

Se aplica el nivel **Restricted** de Pod Security Standards para todos los pods de aplicación, lo que requiere:
- Usuario no-root.
- Sistema de archivos raíz en modo read-only (cuando sea posible).
- Sin capabilities privilegiadas.
- Sin hostNetwork, hostPID, hostIPC.

### Estado del Prototipo

No implementado. Los pods usan configuraciones por defecto sin restricciones.

---

## 8.3. Seguridad en la Aplicación

### Decisión: Keycloak como Identity Provider

Se adopta **Keycloak** como proveedor de identidad (IdP) para autenticación OIDC. La elección se basa en:

- **Self-hosted**: control total sobre la infraestructura de autenticación, sin dependencia de SaaS.
- **Protocolo estándar OIDC/OAuth2**: compatible con cualquier librería de autenticación.
- **Gestión de usuarios integrada**: no requiere otro sistema para administrar cuentas.
- **Federation**: puede conectarse a LDAP/AD corporativo si es necesario.
- **Desplegable en Kubernetes**: Helm chart oficial disponible.

**Alternativas descartadas**:
- *Auth0, Okta*: excelentes productos SaaS pero introducen dependencia externa y coste.
- *Authentik*: alternativa open source más joven, menor madurez.
- *Dex*: solo federación, no gestiona usuarios directamente.

### Flujo de Autenticación

1. Usuario accede al frontend.
2. Frontend detecta que no hay sesión y redirige a Keycloak.
3. Usuario se autentica (usuario/contraseña, o SSO si está configurado).
4. Keycloak devuelve un token JWT (access token + refresh token).
5. Frontend almacena el token y lo envía en cada petición al backend (header `Authorization: Bearer <token>`).
6. Backend valida el JWT contra la clave pública de Keycloak.

### Autorización: RBAC a Nivel de Aplicación

Una vez autenticado, el backend determina los permisos del usuario mediante roles almacenados en el token JWT (claim `roles` configurado en Keycloak):

| Rol | Permisos |
|:----|:---------|
| **viewer** | Ver workflows propios, ver ejecuciones, ver artefactos |
| **editor** | Lo anterior + crear/editar workflows, ejecutar |
| **admin** | Lo anterior + ver todos los workflows, gestionar usuarios |

Los roles se asignan en Keycloak y viajan en el token, evitando consultas adicionales a base de datos en cada petición.

### Decisión: Sealed Secrets para Gestión de Secretos

Se adopta **Sealed Secrets** (Bitnami) para gestionar secretos en un entorno GitOps:

- Los secretos se cifran con una clave pública (solo el controlador en el clúster puede descifrarlos).
- Los secretos cifrados (`SealedSecret`) pueden comitearse en el repositorio de infraestructura.
- El controlador en Kubernetes los convierte en `Secret` estándar al aplicarlos.

**Alternativas descartadas**:
- *HashiCorp Vault*: muy potente pero añade complejidad operativa significativa para el tamaño actual de la plataforma.
- *External Secrets Operator*: requiere un almacén externo (AWS Secrets Manager, Vault), no es self-contained.
- *SOPS*: buena opción, pero Sealed Secrets tiene mejor integración nativa con Kubernetes.

Para una evolución futura hacia multiclúster o requisitos de auditoría avanzados, Vault sería la opción a considerar.

### Protección de Endpoints

- **HTTPS obligatorio**: TLS termina en el Ingress Controller (cert-manager + Let's Encrypt).
- **CORS restringido**: solo el dominio del frontend está permitido.
- **Rate limiting**: configurado en el Ingress (nginx annotations).
- **Validación de entrada**: Pydantic en el backend valida todos los payloads.

### Estado del Prototipo

No implementado. El acceso es abierto, sin autenticación ni autorización.

---

## Resumen de Decisiones

| Área | Herramienta | Justificación |
|:-----|:------------|:--------------|
| Escaneo de imágenes | **Trivy** | Open source, fácil integración, cobertura completa |
| Identity Provider | **Keycloak** | Self-hosted, OIDC estándar, gestión de usuarios incluida |
| Gestión de secretos | **Sealed Secrets** | Compatible con GitOps, sin infraestructura adicional |
| TLS | **cert-manager** + Let's Encrypt | Automatización de certificados |

Estas decisiones son coherentes con el resto de la arquitectura: herramientas open source, desplegables en Kubernetes, y sin dependencias de servicios cloud específicos.
