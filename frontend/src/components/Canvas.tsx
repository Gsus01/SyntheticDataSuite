import { useCallback, useRef } from 'react';
import type { DragEvent } from 'react';
import ReactFlow, {
    addEdge,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    useReactFlow,
} from 'reactflow';
import type { Node } from 'reactflow';
import 'reactflow/dist/style.css';

import {
    nodes as initialNodes,
    edges as initialEdges,
} from './initial-elements.js';

// Importar el generador de nodos
import { generateNodeTypes, getNodeConfigByType } from '../utils/nodeGenerator';

const onInit = (reactFlowInstance: unknown) => {
    console.log('flow loaded:', reactFlowInstance);
};

// Generar automáticamente todos los tipos de nodos
const nodeTypes = generateNodeTypes();

let id = 0;
const getId = () => `dndnode_${id++}`;

const Canvas = () => {
    const reactFlowWrapper = useRef<HTMLDivElement>(null);
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const { screenToFlowPosition } = useReactFlow();

    const onConnect = useCallback(
        (params: Parameters<typeof addEdge>[0]) => setEdges((eds) => addEdge(params, eds)),
        [setEdges]
    );

    const onDragOver = useCallback((event: DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event: DragEvent) => {
            event.preventDefault();

            const type = event.dataTransfer.getData('application/reactflow-type');
            const name = event.dataTransfer.getData('application/reactflow-name');

            // check if the dropped element is valid
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const position = screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });
            
            // Obtener la configuración del nodo para pasar datos adicionales
            const nodeConfig = getNodeConfigByType(type);
            
            const newNode = {
                id: getId(),
                type,
                position,
                data: { 
                    label: name,
                    nodeConfig 
                },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [screenToFlowPosition, setNodes]
    );

    return (
        <div className="reactflow-wrapper flex-1 h-full" ref={reactFlowWrapper}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onInit={onInit}
                onDrop={onDrop}
                onDragOver={onDragOver}
                nodeTypes={nodeTypes}
                fitView
                attributionPosition="top-right"
            >
                <MiniMap
                    nodeStrokeColor={(n: Node) => {
                        const nodeConfig = getNodeConfigByType(n.type || 'default');
                        return nodeConfig?.color || '#1a192b';
                    }}
                    nodeColor={(n: Node) => {
                        if (n.style?.background) return n.style.background as string;
                        return "#fff";
                    }}
                    nodeBorderRadius={2}
                />
                <Controls />
                <Background color="#aaa" gap={16} />
            </ReactFlow>
        </div>
    );
};

export default Canvas;