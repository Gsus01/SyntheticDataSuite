# 1. Introducción y Alcance

## 1.1. Propósito del Documento
Este documento tiene como objetivo definir la arquitectura técnica y funcional de la **Plataforma de Orquestación de Gemelos Digitales**. El propósito principal es establecer las bases de cómo debe construirse un sistema capaz de gestionar, ejecutar y supervisar simulaciones complejas de forma eficiente y escalable.

Es importante aclarar desde el principio que este documento describe la **arquitectura de referencia** completa, es decir, el diseño ideal que tendría el sistema en un entorno de producción real. Sin embargo, actualmente el proyecto se encuentra en una fase de validación técnica. Para esta fase, hemos desarrollado un **prototipo funcional** que implementa el núcleo crítico de esta arquitectura (la ejecución de flujos y la interfaz visual), dejando otros componentes simplificados o pendientes de desarrollo futuro.

A lo largo de los siguientes apartados, detallaremos tanto el diseño final esperado como las soluciones específicas adoptadas en el prototipo actual para validar la viabilidad del proyecto.

## 1.2. Visión General del Sistema
El problema que venimos a resolver es bastante claro: los equipos científicos y de ingeniería crean modelos de datos y simulaciones (gemelos digitales) que son cada vez más complejos. A día de hoy, ejecutar estos modelos implica a menudo lidiar con scripts manuales, configuraciones de máquinas complicadas y procesos que son difíciles de repetir si cambia una sola variable.

Nuestra solución busca eliminar esa fricción. Queremos ofrecer una plataforma donde un usuario pueda centrarse en *qué* quiere simular, y no en *dónde* o *cómo* se ejecuta.

La visión es tener un sistema centralizado que permita:
1.  **Diseñar visualmente** los procesos: Que puedas "dibujar" el flujo de tu simulación conectando cajas (nodos) en una pantalla, donde cada caja hace una tarea específica.
2.  **Ejecutar sin dolores de cabeza**: Cuando le das al botón de "ejecutar", el sistema se encarga por detrás de levantar los contenedores necesarios, pasarles los datos y guardar los resultados.
3.  **Tenerlo todo organizado**: Cada ejecución queda registrada. Sabes qué datos entraron, qué salió y si hubo algún error, sin tener que bucear en logs de servidores dispersos.

Como se mencionará más adelante en el apartado 5, ya contamos con un prototipo operativo de laboratorio que nos ha permitido demostrar que esta visión es técnicamente posible utilizando tecnologías como Kubernetes y Argo Workflows.

## 1.3. Alcance
Para que no haya confusiones sobre qué hace la plataforma y qué no, definimos aquí los límites del sistema.

### ¿Qué incluye el sistema? (Alcance)
El alcance de la plataforma incluye todo lo necesario para la **gestión y orquestación** de los procesos:

*   **El Editor Visual**: La aplicación web donde los usuarios diseñan sus flujos de trabajo arrastrando elementos.
*   **El Motor de Ejecución**: Toda la "fontanería" interna (basada en el prototipo actual) que coge ese diseño visual y lo convierte en órdenes reales para el clúster (crear pods, asignar recursos, etc.).
*   **Gestión de Datos de la Simulación**: El mecanismo para mover ficheros (CSVs, imágenes, modelos) de un paso a otro de la simulación de forma automática.
*   **Control y Monitorización**: Herramientas para ver en tiempo real si una simulación está corriendo, si ha fallado o si ha terminado con éxito.
*   **Gestión de Usuarios (Nivel Arquitectura)**: Diseño de cómo los usuarios acceden y se autentican en la plataforma, aunque el prototipo tenga una versión simplificada de esto.

### ¿Qué queda fuera? (Fuera de Alcance)
Hay cosas de las que la plataforma **no** se hace cargo:

*   **El código interno de los modelos**: La plataforma actúa como un director de orquesta, pero no toca los instrumentos. Si un usuario sube un modelo matemático que calcula mal una trayectoria, es responsabilidad del desarrollador del modelo, no de la plataforma. Nosotros nos encargamos de ejecutar el contenedor, no de validar su física o su matemática interna.
*   **Infraestructura Física**: Asumimos que "por debajo" hay un proveedor de Kubernetes (ya sea AWS, Azure o servidores propios). La instalación del cableado o los servidores físicos no es parte de este proyecto de software.

## 1.4. Definiciones y Acrónimos
Para asegurarnos de que hablamos el mismo idioma a lo largo del documento, aclaramos algunos términos técnicos clave:

*   **K8s (Kubernetes)**: Es el "sistema operativo" del clúster. Es la tecnología base que usamos para manejar los contenedores.
*   **Pod**: En nuestro contexto, imagina que es un "trabajador temporal". Cada vez que un paso del flujo tiene que ejecutarse, se crea un Pod para hacer el trabajo y luego desaparece.
*   **PVC (Persistent Volume Claim)**: Es como un "disco duro externo" que enchufamos a los Pods para que puedan guardar datos de forma permanente, para que no se pierdan cuando el Pod termina.
*   **DAG (Grafo Acíclico Dirigido)**: Es el nombre técnico de los "flujos" o diagramas que dibujan los usuarios. "Acíclico" significa que el flujo siempre avanza y no se queda en bucles infinitos.
*   **Ingress**: Es la puerta de entrada a nuestra aplicación desde fuera del clúster.
*   **Argo Workflows**: Es la herramienta específica que hemos elegido para gestionar los flujos complejos en Kubernetes. Es el motor que mueve los engranajes por debajo.
