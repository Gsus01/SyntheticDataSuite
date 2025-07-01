import { create } from 'zustand';
import {
  Connection,
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  addEdge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
} from 'reactflow';

// Definimos los tipos para el estado del store
export interface RFState {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  addNode: (newNode: Node) => void;
  // Futuras acciones: setNodes, setEdges, updateNodeConfig, etc.
}

const useWorkflowStore = create<RFState>((set, get) => ({
  nodes: [
    { id: 'store-1', type: 'input', data: { label: 'Start (from Store)' }, position: { x: 250, y: 5 } },
    { id: 'store-2', data: { label: 'Default (from Store)' }, position: { x: 100, y: 100 } },
  ],
  edges: [{ id: 'e-store-1-2', source: 'store-1', target: 'store-2', animated: true }],

  onNodesChange: (changes: NodeChange[]) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
    });
  },

  onEdgesChange: (changes: EdgeChange[]) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },

  onConnect: (connection: Connection) => {
    set({
      edges: addEdge(connection, get().edges),
    });
  },

  addNode: (newNode: Node) => {
    set({
      nodes: [...get().nodes, newNode],
    });
  }
  // Implementar más acciones según sea necesario
}));

export default useWorkflowStore;
