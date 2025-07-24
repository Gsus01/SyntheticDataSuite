import React from 'react';
import { nodeTypes, categoryConfig } from '../data/nodeTypes';
import type { NodePaletteItem } from '../data/nodeTypes';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  const onDragStart = (event: React.DragEvent, item: NodePaletteItem) => {
    event.dataTransfer.setData('application/reactflow-type', item.type);
    event.dataTransfer.setData('application/reactflow-name', item.name);
    event.dataTransfer.setData('application/reactflow-category', item.category);
    event.dataTransfer.effectAllowed = 'move';
  };

  // Agrupar nodos por categoría
  const groupedNodes = Object.entries(categoryConfig).map(([category, config]) => ({
    category: category as keyof typeof categoryConfig,
    config,
    nodes: nodeTypes.filter(node => node.category === category)
  }));

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h3>Component Palette</h3>
      </div>
      
      <div className="sidebar-content">
        {groupedNodes.map(({ category, config, nodes }) => (
          <div key={category} className="category-section">
            <h4 
              className="category-title"
              style={{ color: config.color }}
            >
              {config.label} ({nodes.length})
            </h4>
            
            <div className="nodes-list">
              {nodes.map((node) => (
                <div
                  key={node.id}
                  className="node-item"
                  draggable
                  onDragStart={(event) => onDragStart(event, node)}
                  style={{ borderLeftColor: node.color }}
                  title={node.description}
                >
                  <div className="node-name">{node.name}</div>
                  <div className="node-description">{node.description}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      

    </div>
  );
};

export default Sidebar;
