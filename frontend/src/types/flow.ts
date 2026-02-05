// export type NodeArtifact = {
//   bucket: string;
//   key: string;
//   size: number;
//   contentType?: string | null;
//   originalFilename?: string | null;
// };

// export type WorkflowNodeRuntimeStatus = {
//   slug: string;
//   phase?: string | null;
//   type?: string | null;
//   id?: string | null;
//   displayName?: string | null;
//   message?: string | null;
//   progress?: string | null;
//   startedAt?: string | null;
//   finishedAt?: string | null;
// };

// export type NodeArtifactPort = {
//   name: string;
//   path?: string | null;
// };

// export type FlowNodePorts = {
//   inputs: NodeArtifactPort[];
//   outputs: NodeArtifactPort[];
// };

// export type FlowNodeData = {
//   label: string;
//   tone?: string;
//   templateName?: string;
//   parameterKeys?: string[];
//   parameterDefaults?: Record<string, unknown>;
//   parameters?: Record<string, unknown>;
//   uploadedArtifact?: NodeArtifact;
//   runtimeStatus?: WorkflowNodeRuntimeStatus;
//   artifactPorts?: FlowNodePorts;
// };
import { Node, Edge } from 'reactflow';

// --- Tus tipos existentes ---

export type NodeArtifact = {
  bucket: string;
  key: string;
  size: number;
  contentType?: string | null;
  originalFilename?: string | null;
};

export type WorkflowNodeRuntimeStatus = {
  slug: string;
  phase?: string | null;
  type?: string | null;
  id?: string | null;
  displayName?: string | null;
  message?: string | null;
  progress?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
};

export type NodeArtifactPort = {
  name: string;
  path?: string | null;
};

export type FlowNodePorts = {
  inputs: NodeArtifactPort[];
  outputs: NodeArtifactPort[];
};

export type FlowNodeData = {
  label: string;
  tone?: string;
  templateName?: string;
  parameterKeys?: string[];
  parameterDefaults?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
  uploadedArtifact?: NodeArtifact;
  runtimeStatus?: WorkflowNodeRuntimeStatus;
  artifactPorts?: FlowNodePorts;
};

// --- LO NUEVO PARA COPY/PASTE ---

export type CopiedFlowData = {
  // Usamos el Genérico <FlowNodeData> para que TS sepa que
  // estos nodos contienen TUS datos (label, tone, etc.)
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  timestamp: number;
};