import nodeConfig from './nodeConfig.json';

// Interfaces para el tipado TypeScript
export interface NodeHandles {
  inputs?: number;
  outputs?: number;
}

export interface NodeParameter {
  name: string;
  type: string;
  default?: any;
  description?: string;
}

export interface NodeConfig {
  parameters?: NodeParameter[];
}

export interface NodePaletteItem {
  id: string;
  type: string;
  name: string;
  description: string;
  category: string;
  color: string;
  icon?: string;
  handles?: NodeHandles;
  config?: NodeConfig;
}

export interface CategoryConfig {
  label: string;
  color: string;
  description: string;
  bgColor: string;
  borderColor: string;
  textColor: string;
}

export interface NodeConfiguration {
  categories: Record<string, CategoryConfig>;
  nodes: NodePaletteItem[];
}

// Cargar configuración desde JSON
const configuration: NodeConfiguration = nodeConfig as NodeConfiguration;

// Exportar las configuraciones
export const nodeTypes: NodePaletteItem[] = configuration.nodes;
export const categoryConfig: Record<string, CategoryConfig> = configuration.categories;

// Funciones auxiliares
export const getNodeConfigByType = (type: string): NodePaletteItem | undefined => {
  return nodeTypes.find(node => node.type === type);
};

export const getNodesByCategory = (category: string): NodePaletteItem[] => {
  return nodeTypes.filter(node => node.category === category);
};

export const getAllCategories = (): string[] => {
  return Object.keys(categoryConfig);
};
