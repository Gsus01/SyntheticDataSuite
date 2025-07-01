import React, { useCallback, useRef } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  Node,
  ReactFlowInstance, // Necesario para project
} from 'reactflow';
import useWorkflowStore, { RFState } from '../hooks/useWorkflowStore'; // Importar el store

// Selector para obtener solo las partes necesarias del store, optimiza re-renders
const selector = (state: RFState) => ({
  nodes: state.nodes,
  edges: state.edges,
  onNodesChange: state.onNodesChange,
  onEdgesChange: state.onEdgesChange,
  onConnect: state.onConnect,
  addNode: state.addNode,
});

let id = 0; // Simple ID generator para nuevos nodos
const getId = () => `dndnode_${id++}`;

const Canvas: React.FC = () => {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode } = useWorkflowStore(selector);
  const reactFlowWrapper = useRef<HTMLDivElement>(null); // Ref para el contenedor de ReactFlow
  const [reactFlowInstance, setReactFlowInstance] = React.useState<ReactFlowInstance | null>(null);


  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) {
        return;
      }

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow-type');
      const name = event.dataTransfer.getData('application/reactflow-name') || type; // Usar nombre o tipo como etiqueta

      // Comprobar si el tipo es válido (opcional, pero buena práctica)
      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node = {
        id: getId(),
        type, // El tipo de nodo (ej: 'preprocess', 'trainHMM')
        position,
        data: { label: `${name}` }, // Etiqueta inicial del nodo
      };

      addNode(newNode);
    },
    [reactFlowInstance, addNode] // Incluir addNode en las dependencias
  );

  return (
    <div className="flex-grow h-full bg-neutral-200" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onInit={setReactFlowInstance} // Guardar la instancia de ReactFlow
        fitView
        className="bg-white"
      >
        <Controls />
        <MiniMap nodeStrokeWidth={3} zoomable pannable />
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#ccc" />
      </ReactFlow>
    </div>
  );
};

export default Canvas;
