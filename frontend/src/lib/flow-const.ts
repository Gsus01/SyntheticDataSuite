export const DND_MIME = "application/reactflow" as const;

export const NODE_TYPES = {
  nodeInput: "nodeInput",
  nodeDefault: "nodeDefault",
  nodeOutput: "nodeOutput",
} as const;

export type NodeTypeId = typeof NODE_TYPES[keyof typeof NODE_TYPES];

