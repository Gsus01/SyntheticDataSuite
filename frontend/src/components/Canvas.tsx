import { useCallback } from 'react';
import ReactFlow, {
    addEdge,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
} from 'reactflow';
import type { Node } from 'reactflow';
import 'reactflow/dist/style.css';

import {
    nodes as initialNodes,
    edges as initialEdges,
} from './initial-elements.js';

const onInit = (reactFlowInstance: unknown) => {
    console.log('flow loaded:', reactFlowInstance);
};

const Canvas = () => {

    const [nodes,, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    const OnConnect = useCallback(
        (params: Parameters<typeof addEdge>[0]) => setEdges((eds: any[]) => addEdge(params, eds)),
        [setEdges]
    );

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={OnConnect}
            onInit={onInit}
            fitView
            attributionPosition="top-right"
        >
            <MiniMap
                nodeStrokeColor={(n: Node) => {
                    if (n.style?.background) return n.style.background;
                    if (n.type === "input") return "#0041d0";
                    if (n.type === "output") return "#ff0072";
                    if (n.type === "default") return "#1a192b";
                    return "#eee";
                }}
                nodeColor={(n: Node) => {
                    if (n.style?.background) return n.style.background;
                    return "#fff";
                }}
                nodeBorderRadius={2}
            />
            <Controls />
            <Background color="#aaa" gap={16} />
        </ReactFlow>
    );
};

export default Canvas;