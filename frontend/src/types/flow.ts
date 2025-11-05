export type NodeArtifact = {
  bucket: string;
  key: string;
  size: number;
  contentType?: string | null;
  originalFilename?: string | null;
};

export type FlowNodeData = {
  label: string;
  tone?: string;
  templateName?: string;
  parameterKeys?: string[];
  parameterDefaults?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
  uploadedArtifact?: NodeArtifact;
};

