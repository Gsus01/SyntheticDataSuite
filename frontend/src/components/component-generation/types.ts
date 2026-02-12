import type { Node } from "@xyflow/react";
import type { ComponentGenerationDecisionStage } from "@/lib/component-generation";

export type FormState = {
  includeMarkdown: boolean;
  maxCharsPerFile: number;
  provider: "ollama" | "openrouter";
  model: string;
  temperature: number;
  noLlm: boolean;
  noStructuredOutput: boolean;
  ollamaUrl: string;
  openrouterUrl: string;
  openrouterKey: string;
  openrouterProvider: string;
};

export type WorkflowNodeState =
  | "pending"
  | "running"
  | "waiting_decision"
  | "completed"
  | "failed"
  | "canceled";

export type WorkflowGraphNodeData = {
  label: string;
  state: WorkflowNodeState;
  message?: string | null;
};

export type WorkflowGraphNodeType = Node<WorkflowGraphNodeData, "workflowNode">;

export type LogEntry = {
  id: string;
  timestamp?: string;
  level: string;
  source: string;
  line: string;
};

export type LogFilter =
  | "ALL"
  | "ERROR"
  | "WARNING"
  | "INFO"
  | "DEBUG"
  | "STDOUT"
  | "STDERR";

export type KeyValueLine = {
  key: string;
  value: string;
};

export type PlanPortView = {
  name: string;
  path: string;
  role: string;
  extraFields: KeyValueLine[];
};

export type PlanComponentView = {
  name: string;
  title: string;
  type: string;
  description: string;
  inputs: PlanPortView[];
  outputs: PlanPortView[];
  parameters: KeyValueLine[];
  notes: string[];
  extraFields: KeyValueLine[];
};

export type PlanViewModel = {
  rationale: string;
  assumptions: string[];
  components: PlanComponentView[];
  extraFields: KeyValueLine[];
};

export type IntegrationSummaryFile = {
  name: string;
  path: string;
};

export type IntegrationSummaryComponent = {
  name: string;
  title: string;
  version: string;
  type: string;
  image: string;
  description: string;
  files: IntegrationSummaryFile[];
};

export type IntegrationSummaryView = {
  componentCount: number;
  components: IntegrationSummaryComponent[];
};

export type DecisionStage = ComponentGenerationDecisionStage | null;

export type UiNotice = {
  kind: "info" | "success";
  message: string;
};

export type ToggleSwitchProps = {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  className?: string;
  labelClassName?: string;
};

export type GraphEdgeTemplate = {
  id: string;
  source: string;
  target: string;
  label?: string;
  dashed?: boolean;
};

export type NodeTheme = {
  toneClass: string;
  pulseClass: string;
};
