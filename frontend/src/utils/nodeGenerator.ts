import type { ComponentType } from 'react';
import GenericNode from '../components/nodes/GenericNode';
import type { NodePaletteItem } from '../data/nodeTypes';
import { nodeTypes } from '../data/nodeTypes';

// Función para crear un componente de nodo específico
const createNodeComponent = (nodeConfig: NodePaletteItem): ComponentType<any> => {
  return ({ data }: { data: any }) => {
    return GenericNode({ 
      data: { 
        ...data, 
        nodeConfig 
      } 
    });
  };
};

// Genera automáticamente todos los tipos de nodos basándose en la configuración
export const generateNodeTypes = () => {
  const generatedNodeTypes: Record<string, ComponentType<any>> = {};
  
  nodeTypes.forEach((nodeConfig) => {
    generatedNodeTypes[nodeConfig.type] = createNodeComponent(nodeConfig);
  });
  
  return generatedNodeTypes;
};

// Función auxiliar para obtener la configuración de un nodo por tipo
export const getNodeConfigByType = (type: string): NodePaletteItem | undefined => {
  return nodeTypes.find(node => node.type === type);
};
