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

export type FlowNodeData = {
  label: string;
  tone?: string;
  templateName?: string;
  parameterKeys?: string[];
  parameterDefaults?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
  uploadedArtifact?: NodeArtifact;
  runtimeStatus?: WorkflowNodeRuntimeStatus;
};

