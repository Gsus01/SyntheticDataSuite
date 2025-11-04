export type FlowNodeData = {
  label: string;
  tone?: string;
  templateName?: string;
  parameterKeys?: string[];
  parameterDefaults?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
};

