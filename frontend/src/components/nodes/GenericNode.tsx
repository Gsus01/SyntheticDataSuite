import { Handle, Position } from 'reactflow';
import type { NodePaletteItem } from '../../data/nodeTypes';
import { categoryConfig } from '../../data/nodeTypes';

interface GenericNodeProps {
  data: {
    label?: string;
    nodeConfig: NodePaletteItem;
  };
}

const GenericNode = ({ data }: GenericNodeProps) => {
  const { nodeConfig } = data;
  const categoryStyle = categoryConfig[nodeConfig.category];
  
  // Si no hay configuración de categoría, usar valores por defecto
  const defaultStyle = {
    bgColor: 'bg-gray-100',
    borderColor: 'border-gray-200',
    textColor: 'text-gray-800'
  };
  
  const styles = categoryStyle || defaultStyle;
  
  return (
    <div className={`px-4 py-2 shadow-md rounded-md ${styles.bgColor} border-2 ${styles.borderColor} min-w-[150px]`}>
      <div className="flex items-center gap-2">
        {nodeConfig.icon && (
          <span className="text-lg">{nodeConfig.icon}</span>
        )}
        <div className={`text-sm font-bold ${styles.textColor}`}>
          {data.label || nodeConfig.name}
        </div>
      </div>
      
      {nodeConfig.description && (
        <div className={`text-xs ${styles.textColor} opacity-70 mt-1`}>
          {nodeConfig.description}
        </div>
      )}
      
      {/* Handles de entrada */}
      {nodeConfig.handles?.inputs && nodeConfig.handles.inputs > 0 && (
        Array.from({ length: nodeConfig.handles.inputs }, (_, index) => (
          <Handle
            key={`input-${index}`}
            type="target"
            position={Position.Top}
            id={`input-${index}`}
            className="w-3 h-3"
            style={{ 
              backgroundColor: nodeConfig.color,
              left: nodeConfig.handles!.inputs! > 1 ? `${20 + (index * 60)}%` : '50%'
            }}
          />
        ))
      )}
      
      {/* Handles de salida */}
      {nodeConfig.handles?.outputs && nodeConfig.handles.outputs > 0 && (
        Array.from({ length: nodeConfig.handles.outputs }, (_, index) => (
          <Handle
            key={`output-${index}`}
            type="source"
            position={Position.Bottom}
            id={`output-${index}`}
            className="w-3 h-3"
            style={{ 
              backgroundColor: nodeConfig.color,
              left: nodeConfig.handles!.outputs! > 1 ? `${20 + (index * 60)}%` : '50%'
            }}
          />
        ))
      )}
    </div>
  );
};

export default GenericNode;
