# Node Configuration System

Este sistema permite definir nodos de React Flow usando únicamente archivos de configuración JSON, sin necesidad de escribir componentes React específicos para cada tipo de nodo.

## Archivos Principales

### `nodeConfig.json`
Archivo de configuración principal que define todos los nodos y categorías disponibles en la aplicación.

### `nodeConfigTemplate.json`
Plantilla completa con ejemplos y documentación para crear nuevas configuraciones.

### `nodeTypes.ts`
Carga la configuración JSON y exporta las interfaces TypeScript necesarias.

### `components/nodes/GenericNode.tsx`
Componente genérico que renderiza cualquier tipo de nodo basándose en su configuración.

### `utils/nodeGenerator.ts`
Genera automáticamente todos los tipos de nodos de React Flow basándose en la configuración.

## Estructura de la Configuración

### Categorías
```json
{
  "categories": {
    "category_name": {
      "label": "Display Name",
      "color": "#hex_color",
      "description": "Category description",
      "bgColor": "bg-tailwind-class",
      "borderColor": "border-tailwind-class", 
      "textColor": "text-tailwind-class"
    }
  }
}
```

### Nodos
```json
{
  "nodes": [
    {
      "id": "unique-node-id",
      "type": "uniqueNodeType",
      "name": "Display Name",
      "description": "Node description",
      "category": "category_name",
      "color": "#hex_color",
      "icon": "🔧",
      "handles": {
        "inputs": 1,
        "outputs": 1
      },
      "config": {
        "parameters": [
          {
            "name": "param_name",
            "type": "string|number|boolean",
            "default": "default_value",
            "description": "Parameter description"
          }
        ]
      }
    }
  ]
}
```

## Propiedades de los Nodos

### Obligatorias
- **id**: Identificador único del nodo
- **type**: Tipo usado por React Flow (debe ser único)
- **name**: Nombre mostrado en la sidebar y en el nodo
- **description**: Descripción mostrada en tooltips
- **category**: Debe coincidir con una categoría definida
- **color**: Color hexadecimal del nodo

### Opcionales
- **icon**: Emoji o carácter Unicode para mostrar en el nodo
- **handles**: Configuración de conectores de entrada y salida
  - **inputs**: Número de conectores de entrada (0 para nodos fuente)
  - **outputs**: Número de conectores de salida (0 para nodos sumidero)
- **config**: Configuración avanzada del nodo
  - **parameters**: Array de parámetros configurables

## Cómo Agregar un Nuevo Nodo

1. **Abrir `nodeConfig.json`**

2. **Agregar el nodo al array `nodes`**:
```json
{
  "id": "my-new-node",
  "type": "myNewNodeType",
  "name": "My New Node",
  "description": "This is my new node",
  "category": "preprocessing",
  "color": "#3b82f6",
  "icon": "⚙️",
  "handles": {
    "inputs": 1,
    "outputs": 1
  }
}
```

3. **Guardar el archivo**

4. **El nodo aparecerá automáticamente** en la sidebar y será funcional en React Flow

## Cómo Agregar una Nueva Categoría

1. **Agregar la categoría al objeto `categories`**:
```json
"my_category": {
  "label": "My Category",
  "color": "#ff6b6b",
  "description": "My custom category",
  "bgColor": "bg-red-100",
  "borderColor": "border-red-200",
  "textColor": "text-red-800"
}
```

2. **Usar la categoría en los nodos**:
```json
{
  "category": "my_category",
  // ... resto de propiedades del nodo
}
```

## Tipos de Nodos Especiales

### Nodo Fuente (Sin entradas)
```json
{
  "handles": {
    "inputs": 0,
    "outputs": 1
  }
}
```

### Nodo Sumidero (Sin salidas)
```json
{
  "handles": {
    "inputs": 1,
    "outputs": 0
  }
}
```

### Nodo con Múltiples Conectores
```json
{
  "handles": {
    "inputs": 3,
    "outputs": 2
  }
}
```

## Colores Recomendados

- **Preprocessing**: `#3b82f6` (azul)
- **Training**: `#8b5cf6` (púrpura) 
- **Generation**: `#10b981` (verde)
- **Evaluation**: `#f59e0b` (amarillo)
- **Utilities**: `#6b7280` (gris)

## Iconos Recomendados

Usa emojis Unicode para mejor compatibilidad:
- 🔧 Preprocessing
- 🧠 Training/AI
- ✨ Generation
- 📊 Analysis
- 📁 Data Sources
- 💾 Data Sinks
- 🔍 Search/Filter
- ⚡ Performance
- 🔄 Transformation

## Troubleshooting

### El nodo no aparece en la sidebar
- Verifica que el JSON sea válido
- Asegúrate de que la categoría existe
- Revisa que el `id` y `type` sean únicos

### Error de tipo TypeScript
- Ejecuta `npm run dev` para recompilar
- Verifica que las importaciones estén correctas

### El nodo se ve mal
- Revisa las clases de Tailwind CSS
- Asegúrate de que los colores sean hexadecimales válidos
- Verifica que las propiedades `bgColor`, `borderColor`, `textColor` usen clases válidas de Tailwind

## Ventajas del Sistema

1. **Sin código**: Agregar nodos sin escribir React
2. **Escalable**: Soporta cientos de tipos de nodos
3. **Mantenible**: Un solo lugar para configurar nodos
4. **Flexible**: Configuración rica con iconos, colores, etc.
5. **Tipado**: TypeScript para verificación de tipos
6. **Estándar**: Configuración JSON estándar
