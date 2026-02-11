import type { FormState, GraphEdgeTemplate } from "./types";

export const DEFAULT_FORM: FormState = {
  includeMarkdown: false,
  maxCharsPerFile: 200000,
  provider: "ollama",
  model: "qwen3:14b",
  temperature: 0,
  noLlm: false,
  noStructuredOutput: false,
  ollamaUrl: "http://localhost:11434",
  openrouterUrl: "https://openrouter.ai/api/v1",
  openrouterKey: "",
  openrouterProvider: "",
};

export const COMPONENT_GENERATION_FORM_STORAGE_KEY =
  "syntheticDataSuite.componentGeneration.form.v1";

export const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled"]);
export const MAX_EVENT_ENTRIES = 500;
export const MAX_LOG_ENTRIES = 5000;

export const GRAPH_LAYOUT = [
  { id: "load", label: "load", x: 40, y: 120 },
  { id: "analyst", label: "analyst", x: 260, y: 120 },
  { id: "hitl", label: "hitl", x: 480, y: 120 },
  { id: "developer", label: "developer", x: 700, y: 120 },
  { id: "tester", label: "tester", x: 920, y: 120 },
  { id: "integration", label: "integration", x: 1140, y: 120 },
  { id: "repair", label: "repair", x: 920, y: 300 },
] as const;

export const GRAPH_EDGES_BASE: ReadonlyArray<GraphEdgeTemplate> = [
  { id: "load-analyst", source: "load", target: "analyst" },
  { id: "analyst-hitl", source: "analyst", target: "hitl" },
  {
    id: "hitl-analyst",
    source: "hitl",
    target: "analyst",
    label: "revise",
    dashed: true,
  },
  { id: "hitl-developer", source: "hitl", target: "developer" },
  { id: "developer-tester", source: "developer", target: "tester" },
  {
    id: "tester-repair",
    source: "tester",
    target: "repair",
    label: "needs_fix",
    dashed: true,
  },
  { id: "repair-tester", source: "repair", target: "tester" },
  { id: "tester-integration", source: "tester", target: "integration" },
] as const;
