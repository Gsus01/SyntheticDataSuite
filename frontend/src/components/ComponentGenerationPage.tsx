"use client";

import Link from "next/link";
import React from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Edge,
  Handle,
  MarkerType,
  MiniMap,
  type Node,
  type NodeTypes,
  type NodeProps,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  cancelAllComponentGenerationRuns,
  cancelComponentGenerationRun,
  createComponentGenerationRun,
  fetchComponentGenerationRun,
  type ComponentGenerationRunEvent,
  type ComponentGenerationRunSnapshot,
  subscribeComponentGenerationEvents,
  submitComponentGenerationDecision,
} from "@/lib/component-generation";
import { useTheme } from "@/lib/theme-context";

type FormState = {
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

type WorkflowNodeState =
  | "pending"
  | "running"
  | "waiting_decision"
  | "completed"
  | "failed"
  | "canceled";

type WorkflowGraphNodeData = {
  label: string;
  state: WorkflowNodeState;
  message?: string | null;
};

type WorkflowGraphNodeType = Node<WorkflowGraphNodeData, "workflowNode">;

type LogEntry = {
  id: string;
  timestamp?: string;
  level: string;
  source: string;
  line: string;
};

type LogFilter = "ALL" | "ERROR" | "WARNING" | "INFO" | "DEBUG" | "STDOUT" | "STDERR";

type PersistedFormState = Partial<FormState> & {
  rememberOpenrouterKey?: boolean;
};

type KeyValueLine = {
  key: string;
  value: string;
};

type PlanPortView = {
  name: string;
  path: string;
  role: string;
  extraFields: KeyValueLine[];
};

type PlanComponentView = {
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

type PlanViewModel = {
  rationale: string;
  assumptions: string[];
  components: PlanComponentView[];
  extraFields: KeyValueLine[];
};

type PlanComponent = {
  name: string;
  type: string;
  description?: string;
  inputs?: Record<string, string>;
  outputs?: Record<string, string>;
  dependencies?: string[];
  [key: string]: unknown;
};

type UiNotice = {
  kind: "info" | "success";
  message: string;
};

type ToggleSwitchProps = {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  className?: string;
  labelClassName?: string;
};

const DEFAULT_FORM: FormState = {
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

const COMPONENT_GENERATION_FORM_STORAGE_KEY =
  "syntheticDataSuite.componentGeneration.form.v1";

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled"]);
const MAX_EVENT_ENTRIES = 500;
const MAX_LOG_ENTRIES = 5000;

const GRAPH_LAYOUT = [
  { id: "load", label: "load", x: 40, y: 120 },
  { id: "analyst", label: "analyst", x: 260, y: 120 },
  { id: "hitl", label: "hitl", x: 480, y: 120 },
  { id: "developer", label: "developer", x: 700, y: 120 },
  { id: "tester", label: "tester", x: 920, y: 120 },
  { id: "integration", label: "integration", x: 1140, y: 120 },
  { id: "repair", label: "repair", x: 920, y: 300 },
] as const;

type GraphEdgeTemplate = {
  id: string;
  source: string;
  target: string;
  label?: string;
  dashed?: boolean;
};

const GRAPH_EDGES_BASE: ReadonlyArray<GraphEdgeTemplate> = [
  { id: "load-analyst", source: "load", target: "analyst" },
  { id: "analyst-hitl", source: "analyst", target: "hitl" },
  { id: "hitl-analyst", source: "hitl", target: "analyst", label: "revise", dashed: true },
  { id: "hitl-developer", source: "hitl", target: "developer" },
  { id: "developer-tester", source: "developer", target: "tester" },
  { id: "tester-repair", source: "tester", target: "repair", label: "needs_fix", dashed: true },
  { id: "repair-tester", source: "repair", target: "tester" },
  { id: "tester-integration", source: "tester", target: "integration" },
] as const;

const HITL_COMPONENT_ACCENTS = [
  "22 163 74",
  "37 99 235",
  "249 115 22",
  "244 63 94",
  "168 85 247",
  "14 165 233",
] as const;

function hitlComponentAccent(index: number): string {
  return HITL_COMPONENT_ACCENTS[index % HITL_COMPONENT_ACCENTS.length];
}

function ToggleSwitch({
  label,
  checked,
  onChange,
  className = "",
  labelClassName = "",
}: ToggleSwitchProps) {
  return (
    <label
      className={`cg-toggle-row flex cursor-pointer items-center justify-between gap-3 ${className}`}
    >
      <span className={labelClassName}>{label}</span>
      <span className="relative inline-flex items-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
          className="peer sr-only"
        />
        <span className="h-5 w-10 rounded-full border border-slate-600 bg-slate-800 transition-colors duration-200 peer-checked:border-[rgb(var(--cg-accent-strong))] peer-checked:bg-[rgb(var(--cg-accent))] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-[rgb(var(--cg-accent)/0.35)] peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-[rgb(var(--cg-surface))]" />
        <span className="pointer-events-none absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-slate-100 shadow transition-transform duration-200 peer-checked:translate-x-5" />
      </span>
    </label>
  );
}

function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
}

function runStatusBadge(status: string): string {
  if (status === "queued") return "cg-pill-neutral";
  if (status === "running") return "cg-pill-running";
  if (status === "waiting_decision") return "cg-pill-waiting";
  if (status === "succeeded") return "cg-pill-success";
  if (status === "failed") return "cg-pill-danger";
  if (status === "canceled") return "cg-pill-canceled";
  return "cg-pill-neutral";
}

function getEventTypeInfo(eventType: string): { icon: string; colorClass: string; bgClass: string } {
  switch (eventType) {
    case "run_queued":
      return { icon: "⏳", colorClass: "cg-event-tone-neutral", bgClass: "cg-event-bg-neutral" };
    case "run_started":
      return { icon: "🚀", colorClass: "cg-event-tone-running", bgClass: "cg-event-bg-running" };
    case "node_started":
      return { icon: "▶️", colorClass: "cg-event-tone-running", bgClass: "cg-event-bg-running" };
    case "node_completed":
      return { icon: "✅", colorClass: "cg-event-tone-success", bgClass: "cg-event-bg-success" };
    case "node_failed":
      return { icon: "❌", colorClass: "cg-event-tone-danger", bgClass: "cg-event-bg-danger" };
    case "plan_proposed":
      return { icon: "📋", colorClass: "cg-event-tone-plan", bgClass: "cg-event-bg-plan" };
    case "waiting_decision":
      return { icon: "⚖️", colorClass: "cg-event-tone-waiting", bgClass: "cg-event-bg-waiting" };
    case "decision_made":
      return { icon: "🗳️", colorClass: "cg-event-tone-plan", bgClass: "cg-event-bg-plan" };
    case "resumed":
      return { icon: "⏯️", colorClass: "cg-event-tone-running", bgClass: "cg-event-bg-running" };
    case "run_finished":
      return { icon: "🏁", colorClass: "cg-event-tone-success", bgClass: "cg-event-bg-success" };
    case "run_failed":
      return { icon: "🚨", colorClass: "cg-event-tone-danger", bgClass: "cg-event-bg-danger" };
    case "run_canceled":
      return { icon: "🛑", colorClass: "cg-event-tone-waiting", bgClass: "cg-event-bg-waiting" };
    case "log_line":
      return { icon: "📝", colorClass: "cg-event-tone-neutral", bgClass: "cg-event-bg-neutral" };
    default:
      return { icon: "•", colorClass: "cg-event-tone-neutral", bgClass: "cg-event-bg-neutral" };
  }
}

function hasDetailedPayload(event: ComponentGenerationRunEvent): boolean {
  if (event.type === "plan_proposed" || event.type === "waiting_decision") return true;
  if (!event.payload) return false;
  const skipFields = ["node", "seq", "timestamp"];
  return Object.keys(event.payload).some((k) => !skipFields.includes(k));
}

function eventSummary(event: ComponentGenerationRunEvent): string {
  const payload = event.payload || {};
  switch (event.type) {
    case "run_queued":
      return "Execution queued";
    case "run_started":
      return "Agent execution started";
    case "node_started":
      return `Starting step: ${payload.node || "unknown"}`;
    case "node_completed":
      return `Step ${payload.node || "unknown"} completed successfully`;
    case "node_failed":
      return `Step ${payload.node || "unknown"} failed: ${payload.error || "Unknown error"}`;
    case "plan_proposed":
      return "Agent proposed a generation plan";
    case "waiting_decision":
      return "Awaiting human-in-the-loop approval";
    case "decision_made":
      return `Decision submitted: ${payload.approved ? "Approved" : "Revision requested"}`;
    case "resumed":
      return "Execution resumed after decision";
    case "run_finished":
      return "Agent workflow completed successfully";
    case "run_failed":
      return `Workflow execution failed: ${payload.error || "Unknown error"}`;
    case "run_canceled":
      return "Workflow execution interrupted by user";
    case "log_line":
      return (payload.line as string) || "Log message";
    default:
      return event.type.replace(/_/g, " ");
  }
}

function parseSnapshotLogLine(rawLine: string): Omit<LogEntry, "id"> {
  const line = rawLine ?? "";
  const match = line.match(/^\[([A-Z]+)\]\s+([^:]+):\s?(.*)$/);
  if (!match) {
    return {
      line,
      level: "LOG",
      source: "snapshot",
    };
  }
  return {
    line: match[3] || "",
    level: (match[1] || "LOG").toUpperCase(),
    source: match[2] || "snapshot",
  };
}

function levelClass(level: string): string {
  const normalized = level.toUpperCase();
  if (normalized === "ERROR" || normalized === "CRITICAL") return "cg-log-level-danger";
  if (normalized === "WARNING" || normalized === "WARN") return "cg-log-level-warning";
  if (normalized === "DEBUG") return "cg-log-level-running";
  if (normalized === "STDERR") return "cg-log-level-danger";
  if (normalized === "STDOUT") return "cg-log-level-success";
  return "cg-log-level-default";
}

function shouldIncludeLog(entry: LogEntry, filter: LogFilter): boolean {
  if (filter === "ALL") return true;
  const level = entry.level.toUpperCase();
  if (filter === "WARNING") return level === "WARNING" || level === "WARN";
  return level === filter;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function parsePersistedFormState(
  rawValue: unknown
): { form: FormState; rememberOpenrouterKey: boolean } | null {
  const stored = asRecord(rawValue);
  if (!stored) return null;

  const rememberOpenrouterKey =
    typeof stored.rememberOpenrouterKey === "boolean"
      ? stored.rememberOpenrouterKey
      : false;

  const maxCharsRaw = stored.maxCharsPerFile;
  const maxCharsPerFile =
    typeof maxCharsRaw === "number" &&
      Number.isFinite(maxCharsRaw) &&
      maxCharsRaw > 0
      ? Math.trunc(maxCharsRaw)
      : DEFAULT_FORM.maxCharsPerFile;

  const temperatureRaw = stored.temperature;
  const temperature =
    typeof temperatureRaw === "number" && Number.isFinite(temperatureRaw)
      ? temperatureRaw
      : DEFAULT_FORM.temperature;

  const form: FormState = {
    includeMarkdown:
      typeof stored.includeMarkdown === "boolean"
        ? stored.includeMarkdown
        : DEFAULT_FORM.includeMarkdown,
    maxCharsPerFile,
    provider: stored.provider === "openrouter" ? "openrouter" : "ollama",
    model:
      typeof stored.model === "string" && stored.model.trim()
        ? stored.model
        : DEFAULT_FORM.model,
    temperature,
    noLlm: typeof stored.noLlm === "boolean" ? stored.noLlm : DEFAULT_FORM.noLlm,
    noStructuredOutput:
      typeof stored.noStructuredOutput === "boolean"
        ? stored.noStructuredOutput
        : DEFAULT_FORM.noStructuredOutput,
    ollamaUrl:
      typeof stored.ollamaUrl === "string" && stored.ollamaUrl.trim()
        ? stored.ollamaUrl
        : DEFAULT_FORM.ollamaUrl,
    openrouterUrl:
      typeof stored.openrouterUrl === "string" && stored.openrouterUrl.trim()
        ? stored.openrouterUrl
        : DEFAULT_FORM.openrouterUrl,
    openrouterKey:
      rememberOpenrouterKey && typeof stored.openrouterKey === "string"
        ? stored.openrouterKey
        : "",
    openrouterProvider:
      typeof stored.openrouterProvider === "string"
        ? stored.openrouterProvider
        : DEFAULT_FORM.openrouterProvider,
  };

  return { form, rememberOpenrouterKey };
}

function stringifyFieldValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === null || value === undefined) return "null";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function collectExtraFields(
  value: Record<string, unknown>,
  knownFields: string[]
): KeyValueLine[] {
  const known = new Set(knownFields);
  return Object.entries(value)
    .filter(([key]) => !known.has(key))
    .map(([key, raw]) => ({ key, value: stringifyFieldValue(raw) }));
}

function parsePortForDisplay(rawPort: unknown): PlanPortView {
  const port = asRecord(rawPort) || {};
  const nameRaw = typeof port.name === "string" ? port.name.trim() : "";
  const pathRaw = typeof port.path === "string" ? port.path.trim() : "";
  const roleRaw = typeof port.role === "string" ? port.role.trim() : "";
  return {
    name: nameRaw || "(sin nombre)",
    path: pathRaw || "(sin path)",
    role: roleRaw || "data",
    extraFields: collectExtraFields(port, ["name", "path", "role"]),
  };
}

function parseComponentForDisplay(rawComponent: unknown): PlanComponentView {
  const component = asRecord(rawComponent) || {};
  const inputPorts = Array.isArray(component.inputs) ? component.inputs : [];
  const outputPorts = Array.isArray(component.outputs) ? component.outputs : [];
  const parameterDefaults = asRecord(component.parameters_defaults) || {};
  const notesRaw = Array.isArray(component.notes) ? component.notes : [];

  return {
    name:
      typeof component.name === "string" && component.name.trim()
        ? component.name
        : "(sin nombre)",
    title:
      typeof component.title === "string" && component.title.trim()
        ? component.title
        : "(sin título)",
    type:
      typeof component.type === "string" && component.type.trim()
        ? component.type
        : "(sin tipo)",
    description:
      typeof component.description === "string" && component.description.trim()
        ? component.description
        : "(sin descripción)",
    inputs: inputPorts.map(parsePortForDisplay),
    outputs: outputPorts.map(parsePortForDisplay),
    parameters: Object.entries(parameterDefaults).map(([key, raw]) => ({
      key,
      value: stringifyFieldValue(raw),
    })),
    notes: notesRaw.map((item) => stringifyFieldValue(item)),
    extraFields: collectExtraFields(component, [
      "name",
      "title",
      "type",
      "description",
      "inputs",
      "outputs",
      "parameters_defaults",
      "notes",
    ]),
  };
}

function parsePlanForDisplay(
  pendingPlan: Record<string, unknown> | null | undefined
): PlanViewModel {
  const plan = asRecord(pendingPlan) || {};
  const componentsRaw = Array.isArray(plan.components) ? plan.components : [];
  const assumptionsRaw = Array.isArray(plan.assumptions) ? plan.assumptions : [];

  return {
    rationale:
      typeof plan.rationale === "string" && plan.rationale.trim()
        ? plan.rationale
        : "(sin rationale)",
    assumptions: assumptionsRaw.map((item) => stringifyFieldValue(item)),
    components: componentsRaw.map(parseComponentForDisplay),
    extraFields: collectExtraFields(plan, ["rationale", "assumptions", "components"]),
  };
}

function workflowNodeStateFor(
  run: ComponentGenerationRunSnapshot | null,
  nodeId: string
): WorkflowNodeState {
  const rawState = run?.nodeStates?.[nodeId]?.state;
  if (run?.status === "waiting_decision" && nodeId === "hitl") {
    return "waiting_decision";
  }
  if (run?.status === "canceled" && rawState === "running") {
    return "canceled";
  }
  if (
    rawState === "pending" ||
    rawState === "running" ||
    rawState === "completed" ||
    rawState === "failed" ||
    rawState === "canceled"
  ) {
    return rawState;
  }
  return "pending";
}

function edgeColorFor(
  sourceState: WorkflowNodeState,
  targetState: WorkflowNodeState
): string {
  if (sourceState === "failed" || targetState === "failed") return "rgb(var(--cg-edge-danger))";
  if (sourceState === "canceled" || targetState === "canceled") return "rgb(var(--cg-edge-warn))";
  if (targetState === "running" || sourceState === "running") return "rgb(var(--cg-edge-running))";
  if (sourceState === "completed" && targetState === "completed") {
    return "rgb(var(--cg-edge-success))";
  }
  if (targetState === "waiting_decision") return "rgb(var(--cg-edge-warn))";
  return "rgb(var(--cg-edge-idle))";
}

function nodeTheme(state: WorkflowNodeState): {
  toneClass: string;
  pulseClass: string;
} {
  if (state === "running") {
    return {
      toneClass: "cg-node-running",
      pulseClass: "node-animate-running",
    };
  }
  if (state === "waiting_decision") {
    return {
      toneClass: "cg-node-waiting",
      pulseClass: "node-animate-canceled",
    };
  }
  if (state === "completed") {
    return {
      toneClass: "cg-node-completed",
      pulseClass: "node-animate-completed",
    };
  }
  if (state === "failed") {
    return {
      toneClass: "cg-node-failed",
      pulseClass: "node-animate-failed",
    };
  }
  if (state === "canceled") {
    return {
      toneClass: "cg-node-canceled",
      pulseClass: "node-animate-canceled",
    };
  }
  return {
    toneClass: "cg-node-pending",
    pulseClass: "node-animate-pending",
  };
}

function stateLabel(state: WorkflowNodeState): string {
  if (state === "waiting_decision") return "waiting approval";
  return state;
}

function WorkflowGraphNode({ data }: NodeProps<WorkflowGraphNodeType>) {
  const theme = nodeTheme(data.state);
  return (
    <div
      className={`cg-workflow-node ${theme.toneClass} ${theme.pulseClass}`}
    >
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={false}
        className="cg-node-handle"
      />
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={false}
        className="cg-node-handle"
      />
      <div className="cg-node-header">
        <span className="cg-node-label">{data.label}</span>
        <span className="cg-node-state">
          {stateLabel(data.state)}
        </span>
      </div>
      <div className="cg-node-message">
        {data.message && data.message.trim()
          ? data.message
          : "LangGraph step"}
      </div>
    </div>
  );
}

const graphNodeTypes: NodeTypes = {
  workflowNode: WorkflowGraphNode,
};

const FORM_FIELD_CLASS =
  "cg-input w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100";

function updateRunFromEvent(
  previous: ComponentGenerationRunSnapshot,
  event: ComponentGenerationRunEvent
): ComponentGenerationRunSnapshot {
  const payload = event.payload || {};
  const next: ComponentGenerationRunSnapshot = {
    ...previous,
    nodeStates: { ...previous.nodeStates },
    lastSeq: Math.max(previous.lastSeq ?? 0, event.seq ?? 0),
    updatedAt: event.timestamp || previous.updatedAt,
  };

  if (event.type === "run_started") {
    next.status = "running";
  } else if (event.type === "plan_proposed" || event.type === "waiting_decision") {
    next.status = event.type === "waiting_decision" ? "waiting_decision" : next.status;
    const plan = payload.plan;
    if (plan && typeof plan === "object") {
      next.pendingPlan = plan as Record<string, unknown>;
    }
    if (typeof payload.prettyPlan === "string") {
      next.pendingPrettyPlan = payload.prettyPlan;
    }
  } else if (event.type === "resumed") {
    next.status = "running";
  } else if (event.type === "run_finished") {
    next.status = "succeeded";
    const generatedIndex = payload.generatedIndex;
    if (generatedIndex && typeof generatedIndex === "object") {
      next.generatedIndex = generatedIndex as Record<string, Record<string, string>>;
    }
    if (typeof payload.reviewReport === "string") {
      next.reviewReport = payload.reviewReport;
    }
    if (typeof payload.reviewStatus === "string") {
      next.reviewStatus = payload.reviewStatus;
    }
    if (typeof payload.integrationReport === "string") {
      next.integrationReport = payload.integrationReport;
    }
  } else if (event.type === "run_failed") {
    next.status = "failed";
    if (typeof payload.error === "string") {
      next.error = payload.error;
    }
  } else if (event.type === "run_canceled") {
    next.status = "canceled";
    next.error = "Run canceled by user.";
  } else if (
    event.type === "node_started" ||
    event.type === "node_completed" ||
    event.type === "node_failed"
  ) {
    const nodeName =
      typeof payload.node === "string" && payload.node.trim()
        ? payload.node.trim()
        : "";
    if (nodeName) {
      const current = next.nodeStates[nodeName] || {
        state: "pending",
        startedAt: null,
        finishedAt: null,
        message: null,
      };
      if (event.type === "node_started") {
        next.nodeStates[nodeName] = {
          ...current,
          state: "running",
          startedAt: event.timestamp,
          message: null,
        };
      } else if (event.type === "node_completed") {
        next.nodeStates[nodeName] = {
          ...current,
          state: "completed",
          finishedAt: event.timestamp,
        };
      } else if (event.type === "node_failed") {
        next.nodeStates[nodeName] = {
          ...current,
          state: "failed",
          finishedAt: event.timestamp,
          message:
            typeof payload.error === "string" && payload.error.trim()
              ? payload.error
              : "Node failed",
        };
      }
    }
  }

  next.awaitingDecision = next.status === "waiting_decision";
  next.canCancel = !isTerminal(next.status);
  return next;
}

export default function ComponentGenerationPage() {
  const { isDark, toggleTheme } = useTheme();
  const [selectedFiles, setSelectedFiles] = React.useState<File[]>([]);
  const [form, setForm] = React.useState<FormState>(DEFAULT_FORM);
  const [run, setRun] = React.useState<ComponentGenerationRunSnapshot | null>(null);
  const [events, setEvents] = React.useState<ComponentGenerationRunEvent[]>([]);
  const [logEntries, setLogEntries] = React.useState<LogEntry[]>([]);
  const [feedback, setFeedback] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [starting, setStarting] = React.useState(false);
  const [decisionLoading, setDecisionLoading] = React.useState(false);
  const [cancelLoading, setCancelLoading] = React.useState(false);
  const [cancelAllLoading, setCancelAllLoading] = React.useState(false);
  const [streamConnected, setStreamConnected] = React.useState(false);
  const [logFilter, setLogFilter] = React.useState<LogFilter>("ALL");
  const [autoScrollLogs, setAutoScrollLogs] = React.useState(true);
  const [rememberOpenrouterKey, setRememberOpenrouterKey] = React.useState(false);
  const [formStorageReady, setFormStorageReady] = React.useState(false);
  const [expandedEvents, setExpandedEvents] = React.useState<Set<number>>(new Set());
  const [mounted, setMounted] = React.useState(false);
  const [notice, setNotice] = React.useState<UiNotice | null>(null);
  const [confirmCancelAll, setConfirmCancelAll] = React.useState(false);

  const toggleEventExpanded = React.useCallback((seq: number) => {
    setExpandedEvents((prev) => {
      const next = new Set(prev);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });
  }, []);

  const eventSourceRef = React.useRef<EventSource | null>(null);
  const logContainerRef = React.useRef<HTMLDivElement | null>(null);
  const nextLogIdRef = React.useRef(1);
  const noticeTimeoutRef = React.useRef<number | null>(null);

  const nextLogId = React.useCallback((): string => {
    const value = nextLogIdRef.current;
    nextLogIdRef.current += 1;
    return `log-${value}`;
  }, []);

  const closeStream = React.useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStreamConnected(false);
  }, []);

  const showNotice = React.useCallback((kind: UiNotice["kind"], message: string) => {
    setNotice({ kind, message });
    if (noticeTimeoutRef.current !== null) {
      window.clearTimeout(noticeTimeoutRef.current);
    }
    noticeTimeoutRef.current = window.setTimeout(() => {
      setNotice(null);
      noticeTimeoutRef.current = null;
    }, 6500);
  }, []);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    return () => {
      if (noticeTimeoutRef.current !== null) {
        window.clearTimeout(noticeTimeoutRef.current);
      }
    };
  }, []);

  React.useEffect(() => {
    try {
      const rawStoredState = window.localStorage.getItem(
        COMPONENT_GENERATION_FORM_STORAGE_KEY
      );
      if (!rawStoredState) return;

      const parsed = JSON.parse(rawStoredState) as PersistedFormState;
      const loadedState = parsePersistedFormState(parsed);
      if (!loadedState) return;

      setForm(loadedState.form);
      setRememberOpenrouterKey(loadedState.rememberOpenrouterKey);
    } catch {
      // Ignore malformed browser-local config.
    } finally {
      setFormStorageReady(true);
    }
  }, []);

  React.useEffect(() => {
    if (!formStorageReady) return;
    try {
      const payload: PersistedFormState = {
        ...form,
        openrouterKey: rememberOpenrouterKey ? form.openrouterKey : "",
        rememberOpenrouterKey,
      };
      window.localStorage.setItem(
        COMPONENT_GENERATION_FORM_STORAGE_KEY,
        JSON.stringify(payload)
      );
    } catch {
      // Ignore storage errors (quota/privacy mode).
    }
  }, [form, rememberOpenrouterKey, formStorageReady]);

  const hydrateLogsFromSnapshot = React.useCallback(
    (tail?: string[]) => {
      if (!tail || !tail.length) return [];
      return tail.map((line) => {
        const parsed = parseSnapshotLogLine(line);
        return {
          id: nextLogId(),
          level: parsed.level,
          source: parsed.source,
          line: parsed.line,
        } satisfies LogEntry;
      });
    },
    [nextLogId]
  );

  const appendLogEntry = React.useCallback((entry: LogEntry) => {
    setLogEntries((prev) => {
      const next = [...prev, entry];
      if (next.length > MAX_LOG_ENTRIES) {
        return next.slice(next.length - MAX_LOG_ENTRIES);
      }
      return next;
    });
  }, []);

  const refreshRun = React.useCallback(
    async (runId: string) => {
      const snapshot = await fetchComponentGenerationRun(runId);
      setRun(snapshot);
      setLogEntries((prev) => {
        if (prev.length === 0 || isTerminal(snapshot.status)) {
          return hydrateLogsFromSnapshot(snapshot.logTail);
        }
        return prev;
      });
      if (isTerminal(snapshot.status)) {
        closeStream();
      }
    },
    [closeStream, hydrateLogsFromSnapshot]
  );

  const connectStream = React.useCallback(
    (runId: string, sinceSeq: number) => {
      closeStream();
      const source = subscribeComponentGenerationEvents(
        runId,
        (event) => {
          setEvents((prev) => {
            const next = [...prev, event];
            if (next.length > MAX_EVENT_ENTRIES) {
              return next.slice(next.length - MAX_EVENT_ENTRIES);
            }
            return next;
          });

          if (event.type === "log_line" && typeof event.payload.line === "string") {
            appendLogEntry({
              id: nextLogId(),
              timestamp: event.timestamp,
              level: (event.payload.level || "INFO").toUpperCase(),
              source: event.payload.source || "worker",
              line: event.payload.line,
            });
          }

          setRun((prev) => (prev ? updateRunFromEvent(prev, event) : prev));

          if (
            event.type === "run_finished" ||
            event.type === "run_failed" ||
            event.type === "run_canceled"
          ) {
            setStreamConnected(false);
            setTimeout(() => {
              closeStream();
            }, 0);
          }
        },
        {
          sinceSeq,
          onOpen: () => setStreamConnected(true),
          onError: () => setStreamConnected(false),
        }
      );
      eventSourceRef.current = source;
    },
    [appendLogEntry, closeStream, nextLogId]
  );

  React.useEffect(() => {
    return () => {
      closeStream();
    };
  }, [closeStream]);

  React.useEffect(() => {
    if (!run?.runId || isTerminal(run.status)) {
      return;
    }
    const timer = setInterval(() => {
      void refreshRun(run.runId).catch(() => {
        // Keep SSE stream as primary source of truth.
      });
    }, 2000);
    return () => clearInterval(timer);
  }, [run?.runId, run?.status, refreshRun]);

  React.useEffect(() => {
    if (!autoScrollLogs) return;
    const container = logContainerRef.current;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
  }, [autoScrollLogs, logEntries]);

  const graphNodes = React.useMemo<WorkflowGraphNodeType[]>(
    () =>
      GRAPH_LAYOUT.map((item) => {
        const state = workflowNodeStateFor(run, item.id);
        const rawMessage = run?.nodeStates?.[item.id]?.message;
        const message =
          state === "waiting_decision"
            ? "Waiting for manual approval"
            : rawMessage;
        return {
          id: item.id,
          type: "workflowNode",
          position: { x: item.x, y: item.y },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          data: {
            label: item.label,
            state,
            message: message || null,
          },
          draggable: false,
          selectable: false,
        };
      }),
    [run]
  );

  const graphEdges = React.useMemo<Edge[]>(() => {
    const stateByNode = new Map<string, WorkflowNodeState>();
    graphNodes.forEach((node) => {
      stateByNode.set(node.id, node.data.state);
    });

    return GRAPH_EDGES_BASE.map((edge) => {
      const sourceState = stateByNode.get(edge.source) || "pending";
      const targetState = stateByNode.get(edge.target) || "pending";
      const stroke = edgeColorFor(sourceState, targetState);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label,
        markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
        style: {
          stroke,
          strokeWidth: 2.2,
          strokeDasharray: edge.dashed ? "6 4" : undefined,
        },
        animated:
          sourceState === "running" ||
          targetState === "running" ||
          targetState === "waiting_decision",
        labelStyle: {
          fill: "rgb(var(--cg-edge-label))",
          fontSize: 11,
          fontWeight: 600,
        },
      };
    });
  }, [graphNodes]);

  const filteredLogs = React.useMemo(
    () => logEntries.filter((entry) => shouldIncludeLog(entry, logFilter)),
    [logEntries, logFilter]
  );

  const planViewModel = React.useMemo(
    () => parsePlanForDisplay(run?.pendingPlan),
    [run?.pendingPlan]
  );

  const startRun = React.useCallback(async () => {
    if (!selectedFiles.length) {
      setError("Selecciona al menos un archivo de entrada.");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const snapshot = await createComponentGenerationRun({
        files: selectedFiles,
        includeMarkdown: form.includeMarkdown,
        maxCharsPerFile: form.maxCharsPerFile,
        provider: form.provider,
        model: form.model,
        temperature: form.temperature,
        noLlm: form.noLlm,
        noStructuredOutput: form.noStructuredOutput,
        ollamaUrl: form.ollamaUrl,
        openrouterUrl: form.openrouterUrl,
        openrouterKey: form.openrouterKey || undefined,
        openrouterProvider: form.openrouterProvider || undefined,
      });
      setRun(snapshot);
      setEvents([]);
      setFeedback("");
      setLogEntries(hydrateLogsFromSnapshot(snapshot.logTail));
      connectStream(snapshot.runId, snapshot.lastSeq);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo iniciar la generación.");
    } finally {
      setStarting(false);
    }
  }, [connectStream, form, hydrateLogsFromSnapshot, selectedFiles]);

  const submitDecision = React.useCallback(
    async (approved: boolean) => {
      if (!run) return;
      setDecisionLoading(true);
      setError(null);
      try {
        const snapshot = await submitComponentGenerationDecision(run.runId, {
          approved,
          feedback: approved ? "" : feedback,
        });
        setRun(snapshot);
        if (approved) {
          setFeedback("");
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "No se pudo enviar la decisión."
        );
      } finally {
        setDecisionLoading(false);
      }
    },
    [feedback, run]
  );

  const cancelRun = React.useCallback(async () => {
    if (!run) return;
    setCancelLoading(true);
    setError(null);
    try {
      const snapshot = await cancelComponentGenerationRun(run.runId);
      setRun(snapshot);
      closeStream();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "No se pudo interrumpir la ejecución."
      );
    } finally {
      setCancelLoading(false);
    }
  }, [closeStream, run]);

  const cancelAllRuns = React.useCallback(async () => {
    setConfirmCancelAll(false);
    setCancelAllLoading(true);
    setError(null);
    try {
      const result = await cancelAllComponentGenerationRuns();
      if (result.canceledCount === 0) {
        showNotice("info", "No había sesiones activas en background.");
      } else {
        showNotice(
          "success",
          `Se cancelaron ${result.canceledCount} sesión(es) en background.`
        );
      }

      if (run && result.canceledRunIds.includes(run.runId)) {
        closeStream();
        try {
          await refreshRun(run.runId);
        } catch {
          // Snapshot refresh is best-effort; stream is already closed.
        }
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "No se pudieron cerrar las sesiones en background."
      );
    } finally {
      setCancelAllLoading(false);
    }
  }, [closeStream, refreshRun, run, showNotice]);

  const resetLocalSettings = React.useCallback(() => {
    setForm(DEFAULT_FORM);
    setRememberOpenrouterKey(false);
    try {
      window.localStorage.removeItem(COMPONENT_GENERATION_FORM_STORAGE_KEY);
    } catch {
      // Ignore storage errors.
    }
  }, []);

  const generatedCount = Object.keys(run?.generatedIndex || {}).length;
  const isAwaitingDecision = Boolean(run?.awaitingDecision);

  return (
    <div className="cg-shell min-h-screen bg-slate-950 text-slate-100">
      <header
        className="cg-header flex h-16 items-center justify-between border-b border-slate-800 bg-slate-950 px-5 cg-reveal cg-reveal-1 overflow-visible"
        style={{ zIndex: 80 }}
      >
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="cg-back-link rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:bg-slate-800"
          >
            Volver al editor
          </Link>
          <div>
            <h1 className="cg-page-title text-sm font-semibold">
              Generación Automática de Componentes
            </h1>
            <p className="cg-page-subtitle text-[11px] text-slate-400">
              Logs estilo CLI, grafo en vivo y revisión HITL completa.
            </p>
          </div>
        </div>
        <div className="relative flex items-center gap-3 cg-status-wrap">
          {run ? (
            <span
              className={`cg-status-pill inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide ${runStatusBadge(
                run.status
              )}`}
            >
              {run.status.replace("_", " ")}
            </span>
          ) : (
            <span className="cg-muted text-xs text-slate-400">Sin ejecución activa</span>
          )}
          <span
            className={`cg-status-pill rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ${streamConnected
              ? "cg-stream-online"
              : "cg-stream-offline"
              }`}
          >
            {streamConnected ? "SSE online" : "SSE offline"}
          </span>
          <button
            type="button"
            onClick={() => {
              setNotice(null);
              setConfirmCancelAll(true);
            }}
            disabled={cancelAllLoading}
            className="cg-btn cg-btn-warning inline-flex items-center justify-center rounded-md px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide disabled:cursor-not-allowed"
            title="Interrumpe todas las ejecuciones activas en background"
          >
            {cancelAllLoading ? "Cerrando..." : "Cerrar sesiones"}
          </button>
          <button
            type="button"
            onClick={toggleTheme}
            className="cg-btn cg-btn-muted inline-flex items-center justify-center rounded-md px-2 py-1 text-sm"
            aria-label="Cambiar tema claro/oscuro"
            suppressHydrationWarning
          >
            {!mounted ? <span className="opacity-0">☀️</span> : isDark ? "☀️" : "🌙"}
          </button>

          {confirmCancelAll ? (
            <div className="absolute right-0 top-full z-[60] mt-2 w-[min(92vw,26rem)]">
              <div className="rounded-xl border border-[rgb(var(--cg-warning)/0.5)] bg-[rgb(var(--cg-warning)/0.2)] px-3 py-2 shadow-xl backdrop-blur-md">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-[12px] leading-snug text-[rgb(var(--cg-text))]">
                    Esto interrumpirá todas las sesiones activas en background.
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setConfirmCancelAll(false)}
                      className="cg-btn cg-btn-muted rounded-md px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide"
                    >
                      Volver
                    </button>
                    <button
                      type="button"
                      onClick={() => void cancelAllRuns()}
                      disabled={cancelAllLoading}
                      className="cg-btn cg-btn-warning rounded-md px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide disabled:cursor-not-allowed"
                    >
                      {cancelAllLoading ? "Cerrando..." : "Sí, cerrar"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {notice && !confirmCancelAll ? (
            <div className="absolute right-0 top-full z-[55] mt-2 w-[min(90vw,24rem)]">
              <div
                className={`flex items-center justify-between gap-3 rounded-xl border px-3 py-2 text-[13px] leading-snug shadow-xl backdrop-blur-md ${
                  notice.kind === "success"
                    ? "border-[rgb(var(--cg-success)/0.5)] bg-[rgb(var(--cg-success)/0.2)] text-[rgb(var(--cg-text))]"
                    : "border-[rgb(var(--cg-border))] bg-[rgb(var(--cg-surface)/0.94)] text-[rgb(var(--cg-text-soft))]"
                }`}
                role="status"
                aria-live="polite"
              >
                <span>{notice.message}</span>
                <button
                  type="button"
                  className="cg-btn cg-btn-muted rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-wide"
                  onClick={() => setNotice(null)}
                >
                  Cerrar
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </header>

      <div className="cg-layout grid h-[calc(100vh-4rem)] grid-cols-1 lg:grid-cols-[360px_1fr]">
        <aside className="cg-sidebar flex min-h-0 flex-col overflow-y-auto border-r border-slate-800 bg-slate-900 p-4 cg-reveal cg-reveal-2">
          <div className="space-y-4">
            <section className="cg-panel rounded-lg border border-slate-800 bg-slate-900/70 p-3 cg-reveal cg-reveal-2">
              <h2 className="cg-section-title text-xs font-semibold uppercase tracking-wide text-slate-300">
                Archivos de entrada
              </h2>
              <label className="cg-dropzone mt-3 flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-slate-700 bg-slate-950/60 px-3 py-4 text-center text-xs text-slate-300 hover:border-slate-600">
                <span className="font-medium">Seleccionar archivos</span>
                <span className="cg-muted mt-1 text-[11px] text-slate-500">
                  .py, .ipynb, .md, etc.
                </span>
                <input
                  type="file"
                  multiple
                  className="hidden"
                  onChange={(event) => {
                    const files = Array.from(event.target.files || []);
                    setSelectedFiles(files);
                  }}
                />
              </label>
              <div className="mt-3 max-h-40 space-y-2 overflow-y-auto">
                {selectedFiles.length === 0 ? (
                  <p className="cg-muted text-[11px] text-slate-500">Sin archivos seleccionados.</p>
                ) : (
                  selectedFiles.map((file) => (
                    <div
                      key={`${file.name}-${file.size}`}
                      className="cg-file-item rounded border border-slate-800 bg-slate-950/60 px-2 py-1.5 text-[11px]"
                    >
                      <div className="truncate font-medium">{file.name}</div>
                      <div className="cg-muted">{formatBytes(file.size)}</div>
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="cg-panel rounded-lg border border-slate-800 bg-slate-900/70 p-3 cg-reveal cg-reveal-3">
              <h2 className="cg-section-title text-xs font-semibold uppercase tracking-wide text-slate-300">
                Configuración
              </h2>
              <div className="mt-3 space-y-3 text-xs">
                <ToggleSwitch
                  label="Incluir markdown"
                  checked={form.includeMarkdown}
                  onChange={(checked) =>
                    setForm((prev) => ({
                      ...prev,
                      includeMarkdown: checked,
                    }))
                  }
                />

                <label className="block">
                  <span className="cg-field-label mb-1 block text-[11px] text-slate-400">
                    Max chars por fichero
                  </span>
                  <input
                    type="number"
                    min={1}
                    value={form.maxCharsPerFile}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        maxCharsPerFile: Number(event.target.value) || 1,
                      }))
                    }
                    className={FORM_FIELD_CLASS}
                  />
                </label>

                <label className="block">
                  <span className="cg-field-label mb-1 block text-[11px] text-slate-400">Provider</span>
                  <select
                    value={form.provider}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        provider: event.target.value as "ollama" | "openrouter",
                      }))
                    }
                    className={FORM_FIELD_CLASS}
                  >
                    <option value="ollama">ollama</option>
                    <option value="openrouter">openrouter</option>
                  </select>
                </label>

                <label className="block">
                  <span className="cg-field-label mb-1 block text-[11px] text-slate-400">Modelo</span>
                  <input
                    type="text"
                    value={form.model}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, model: event.target.value }))
                    }
                    className={FORM_FIELD_CLASS}
                  />
                </label>

                <label className="block">
                  <span className="cg-field-label mb-1 block text-[11px] text-slate-400">
                    Temperatura
                  </span>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={2}
                    value={form.temperature}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        temperature: Number(event.target.value) || 0,
                      }))
                    }
                    className={FORM_FIELD_CLASS}
                  />
                </label>

                <ToggleSwitch
                  label="Desactivar LLM"
                  checked={form.noLlm}
                  onChange={(checked) =>
                    setForm((prev) => ({ ...prev, noLlm: checked }))
                  }
                />

                <ToggleSwitch
                  label="Sin structured output"
                  checked={form.noStructuredOutput}
                  onChange={(checked) =>
                    setForm((prev) => ({
                      ...prev,
                      noStructuredOutput: checked,
                    }))
                  }
                />

                {form.provider === "ollama" ? (
                  <label className="block">
                    <span className="cg-field-label mb-1 block text-[11px] text-slate-400">
                      Ollama URL
                    </span>
                    <input
                      type="text"
                      value={form.ollamaUrl}
                      onChange={(event) =>
                        setForm((prev) => ({ ...prev, ollamaUrl: event.target.value }))
                      }
                      className={FORM_FIELD_CLASS}
                    />
                  </label>
                ) : (
                  <>
                    <label className="block">
                      <span className="cg-field-label mb-1 block text-[11px] text-slate-400">
                        OpenRouter URL
                      </span>
                      <input
                        type="text"
                        value={form.openrouterUrl}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterUrl: event.target.value,
                          }))
                        }
                        className={FORM_FIELD_CLASS}
                      />
                    </label>
                    <label className="block">
                      <span className="cg-field-label mb-1 block text-[11px] text-slate-400">
                        OpenRouter Key
                      </span>
                      <input
                        type="password"
                        value={form.openrouterKey}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterKey: event.target.value,
                          }))
                        }
                        className={FORM_FIELD_CLASS}
                      />
                    </label>
                    <ToggleSwitch
                      label="Recordar API key en este navegador"
                      checked={rememberOpenrouterKey}
                      onChange={(checked) => setRememberOpenrouterKey(checked)}
                      className="rounded border border-slate-800 bg-slate-950/60 px-2 py-1.5"
                      labelClassName="text-[11px]"
                    />
                    <label className="block">
                      <span className="cg-field-label mb-1 block text-[11px] text-slate-400">
                        Provider order
                      </span>
                      <input
                        type="text"
                        value={form.openrouterProvider}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterProvider: event.target.value,
                          }))
                        }
                        placeholder="openai,anthropic"
                        className={FORM_FIELD_CLASS}
                      />
                    </label>
                  </>
                )}

                <button
                  type="button"
                  onClick={resetLocalSettings}
                  className="cg-btn cg-btn-muted w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-[11px] font-medium text-slate-200 hover:bg-slate-800"
                >
                  Restablecer ajustes locales
                </button>
              </div>
            </section>

            <section className="cg-panel rounded-lg border border-slate-800 bg-slate-900/70 p-3 cg-reveal cg-reveal-4">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={startRun}
                  disabled={starting || (run ? !isTerminal(run.status) : false)}
                  className="cg-btn cg-btn-primary inline-flex flex-1 items-center justify-center rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-900/60"
                >
                  {starting ? "Iniciando..." : "Iniciar generación"}
                </button>
                <button
                  type="button"
                  onClick={cancelRun}
                  disabled={!run || !run.canCancel || cancelLoading}
                  className="cg-btn cg-btn-danger inline-flex flex-1 items-center justify-center rounded-md bg-rose-600 px-3 py-2 text-xs font-semibold text-white hover:bg-rose-500 disabled:cursor-not-allowed disabled:bg-rose-900/60"
                >
                  {cancelLoading ? "Interrumpiendo..." : "Interrumpir"}
                </button>
              </div>
              {error ? (
                <p className="cg-error-text mt-2 text-[11px] text-rose-300">{error}</p>
              ) : null}
            </section>
          </div>
        </aside>

        <main className="cg-main flex min-h-0 flex-col">
          <section className="cg-graph-section h-[44%] border-b border-slate-800 px-4 py-3 cg-reveal cg-reveal-2">
            <h2 className="cg-section-title mb-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
              LangGraph en tiempo real
            </h2>
            <div className="cg-graph-surface h-[calc(100%-1.8rem)] rounded-xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950">
              <ReactFlow
                nodes={graphNodes}
                edges={graphEdges}
                nodeTypes={graphNodeTypes}
                fitView
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                panOnDrag
                zoomOnScroll
                className="bg-transparent"
              >
                <Background color="rgb(var(--cg-grid-color))" gap={28} size={1} />
                <MiniMap
                  pannable
                  zoomable
                  className="cg-minimap !bg-slate-950/70 !border !border-slate-700"
                  maskColor="rgba(12, 16, 28, 0.55)"
                />
                <Controls />
              </ReactFlow>
            </div>
          </section>

          <section
            className={`cg-content-grid grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden p-4 ${
              isAwaitingDecision
                ? "xl:grid-cols-[0.62fr_1.38fr]"
                : "xl:grid-cols-[0.95fr_1.45fr]"
            }`}
          >
            <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1 cg-reveal cg-reveal-3">
              <div className="cg-card rounded-xl border border-slate-800 bg-slate-900 p-3">
                <h3 className="cg-card-title text-xs font-semibold uppercase tracking-wide text-slate-300">
                  Estado de ejecución
                </h3>
                {run ? (
                  <div className="mt-2 space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Run ID</span>
                      <span className="font-mono text-[11px] text-slate-200">{run.runId}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Estado</span>
                      <span
                        className={`cg-status-pill rounded px-2 py-0.5 text-[10px] font-semibold uppercase ${runStatusBadge(
                          run.status
                        )}`}
                      >
                        {run.status.replace("_", " ")}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Actualizado</span>
                      <span className="text-slate-200">{formatTimestamp(run.updatedAt)}</span>
                    </div>
                    <div className="pt-1 text-[11px] text-slate-400">
                      Session dir:
                      <div className="mt-1 break-all font-mono text-slate-300">
                        {run.sessionDir}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="cg-muted mt-2 text-xs text-slate-500">
                    Aún no has lanzado ninguna ejecución.
                  </p>
                )}
              </div>

              <div className="cg-card rounded-xl border border-slate-800 bg-slate-900 p-3">
                <h3 className="cg-card-title text-xs font-semibold uppercase tracking-wide text-slate-300">
                  Artefactos y eventos
                </h3>
                {run ? (
                  <div className="mt-2 space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Componentes generados</span>
                      <span className="font-semibold text-slate-100">{generatedCount}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Review status</span>
                      <span className="text-slate-200">{run.reviewStatus || "—"}</span>
                    </div>
                    <div className="cg-code-block max-h-28 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-slate-300">
                      {(run.reviewReport || "(sin reporte)").trim() || "(sin reporte)"}
                    </div>
                    <div className="cg-timeline flex min-h-0 flex-1 flex-col rounded-lg border border-slate-800 bg-slate-950/60 p-2">
                      <div className="mb-2 flex items-center justify-between border-b border-slate-800 pb-1.5">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Timeline de eventos</span>
                        <span className="cg-muted text-[10px] text-slate-500">{events.length} total</span>
                      </div>
                      <div className="space-y-1.5 overflow-y-auto max-h-[400px]">
                        {events.length === 0 ? (
                          <p className="cg-muted text-[11px] text-slate-500 italic">Sin eventos todavía.</p>
                        ) : (
                          events.map((event) => {
                            const typeInfo = getEventTypeInfo(event.type);
                            const isExpanded = expandedEvents.has(event.seq);
                            const canExpand = hasDetailedPayload(event);
                            const payload = event.payload || {};

                            return (
                              <div
                                key={event.seq}
                                className={`cg-event-row rounded border border-slate-800/80 ${typeInfo.bgClass} transition-all duration-200 event-type-${event.type}`}
                              >
                                {/* Header - always visible */}
                                <div
                                  className={`cg-event-row-header flex items-center gap-2 px-2 py-1.5 ${canExpand ? "cursor-pointer hover:bg-white/5" : ""}`}
                                  onClick={() => canExpand && toggleEventExpanded(event.seq)}
                                >
                                  <span className="text-sm" role="img" aria-label={event.type}>
                                    {typeInfo.icon}
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <div className={`font-medium text-[11px] truncate ${typeInfo.colorClass}`}>
                                      {eventSummary(event)}
                                    </div>
                                    <div className="cg-muted flex items-center gap-2 text-[9px] text-slate-500">
                                      <span>#{event.seq}</span>
                                      <span>•</span>
                                      <span>{formatTimestamp(event.timestamp)}</span>
                                    </div>
                                  </div>
                                  {canExpand && (
                                    <span className={`cg-muted transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}>
                                      ▼
                                    </span>
                                  )}
                                </div>

                                {/* Expanded content */}
                                {isExpanded && canExpand && (
                                  <div className="cg-event-expanded border-t border-slate-800/50 px-2 py-1.5 space-y-2">
                                    {/* Show plan components if plan_proposed */}
                                    {Boolean(event.type === "plan_proposed" && payload.plan && typeof payload.plan === "object") && (
                                      <div className="space-y-1.5">
                                        <div className="text-[9px] uppercase text-indigo-400 font-semibold">Componentes sugeridos</div>
                                        {Array.isArray((payload.plan as { components?: unknown }).components) && (
                                          <div className="space-y-1">
                                            {((payload.plan as { components: PlanComponent[] }).components).map((comp, idx) => (
                                              <div key={idx} className="rounded border border-indigo-500/10 bg-indigo-500/5 p-1.5">
                                                <div className="flex items-center gap-1.5">
                                                  <span className="font-semibold text-indigo-200 text-[10px]">{String(comp.name || "Component")}</span>
                                                  <span className="px-1 py-0.5 rounded bg-indigo-900/40 text-[8px] uppercase text-indigo-300 border border-indigo-700/30">
                                                    {String(comp.type || "unknown")}
                                                  </span>
                                                </div>
                                                {Boolean(comp.description) && (
                                                  <div className="cg-muted mt-0.5 text-slate-400 text-[9px] line-clamp-2">
                                                    {String(comp.description)}
                                                  </div>
                                                )}
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    )}

                                    {/* Show inputs/outputs for node events */}
                                    {Boolean(payload.inputs || payload.outputs) && (
                                      <div className="space-y-1.5">
                                        {Boolean(payload.inputs) && (
                                          <div>
                                            <span className="text-[9px] uppercase text-sky-400 font-semibold">Inputs</span>
                                            <pre className="cg-code-block mt-0.5 p-1.5 rounded bg-black/40 text-slate-300 text-[10px] overflow-x-auto max-h-24 overflow-y-auto font-mono">
                                              {JSON.stringify(payload.inputs, null, 2)}
                                            </pre>
                                          </div>
                                        )}
                                        {Boolean(payload.outputs) && (
                                          <div>
                                            <span className="text-[9px] uppercase text-emerald-400 font-semibold">Outputs</span>
                                            <pre className="cg-code-block mt-0.5 p-1.5 rounded bg-black/40 text-slate-300 text-[10px] overflow-x-auto max-h-24 overflow-y-auto font-mono">
                                              {JSON.stringify(payload.outputs, null, 2)}
                                            </pre>
                                          </div>
                                        )}
                                      </div>
                                    )}

                                    {/* Generic payload display for other fields */}
                                    {Object.keys(payload).filter(k => !["node", "inputs", "outputs", "plan", "approved", "error"].includes(k)).length > 0 && (
                                      <div>
                                        <span className="cg-muted text-[9px] uppercase text-slate-500 font-semibold">Detalles</span>
                                        <pre className="cg-code-block mt-0.5 p-1.5 rounded bg-black/40 text-slate-300 text-[10px] overflow-x-auto max-h-32 overflow-y-auto font-mono">
                                          {JSON.stringify(
                                            Object.fromEntries(
                                              Object.entries(payload).filter(([k]) => !["node", "inputs", "outputs", "plan", "approved", "error"].includes(k))
                                            ),
                                            null,
                                            2
                                          )}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                    <div className="cg-code-block max-h-20 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-rose-300">
                      {run.error || "(sin error)"}
                    </div>
                  </div>
                ) : (
                  <p className="cg-muted mt-2 text-xs text-slate-500">Sin artefactos todavía.</p>
                )}
              </div>
            </div>

            <div
              className={`flex min-h-0 flex-col rounded-xl border cg-reveal cg-reveal-4 ${
                isAwaitingDecision
                  ? "border-[rgb(var(--cg-warning)/0.5)] bg-[rgb(var(--cg-surface)/0.94)] shadow-[0_20px_42px_rgb(var(--cg-warning)/0.2)]"
                  : "cg-console border-slate-800 bg-slate-900"
              }`}
            >
              {run?.awaitingDecision ? (
                <div className="min-h-0 flex-1 overflow-y-auto p-4">
                  <div className="rounded-2xl border border-[rgb(var(--cg-warning)/0.56)] bg-[rgb(var(--cg-surface)/0.94)] p-4 shadow-[0_16px_38px_rgb(var(--cg-warning)/0.2)]">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <h3 className="text-[13px] font-semibold uppercase tracking-[0.14em] text-[rgb(var(--cg-warning))]">
                          Revisión HITL requerida
                        </h3>
                        <p className="mt-1 text-[12px] leading-relaxed text-[rgb(var(--cg-text-muted))]">
                          El plan requiere decisión manual. Revisa campos, aprueba o solicita cambios.
                        </p>
                      </div>
                      <span className="rounded-full border border-[rgb(var(--cg-warning)/0.65)] bg-[rgb(var(--cg-warning)/0.25)] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-warning))]">
                        Waiting approval
                      </span>
                    </div>

                    <div className="mt-4 space-y-3">
                      <div className="rounded-lg border border-[rgb(var(--cg-border-soft))] bg-[rgb(var(--cg-bg)/0.58)] p-3">
                        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                          Rationale
                        </div>
                        <p className="mt-1.5 whitespace-pre-wrap text-[12px] leading-relaxed text-[rgb(var(--cg-text-soft))]">
                          {planViewModel.rationale}
                        </p>
                      </div>

                      <div className="rounded-lg border border-[rgb(var(--cg-border-soft))] bg-[rgb(var(--cg-bg)/0.58)] p-3">
                        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                          Assumptions
                        </div>
                        {planViewModel.assumptions.length === 0 ? (
                          <p className="mt-1 text-[12px] text-[rgb(var(--cg-text-muted))]">
                            Sin assumptions.
                          </p>
                        ) : (
                          <ul className="mt-1.5 space-y-1.5 text-[12px] text-[rgb(var(--cg-text-soft))]">
                            {planViewModel.assumptions.map((assumption, index) => (
                              <li
                                key={`assumption-${index}`}
                                className="rounded-md border border-[rgb(var(--cg-border-soft)/0.8)] bg-[rgb(var(--cg-surface)/0.8)] px-2.5 py-1.5"
                              >
                                {assumption}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>

                      <div>
                        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                          Components ({planViewModel.components.length})
                        </div>
                        {planViewModel.components.length === 0 ? (
                          <div className="rounded-lg border border-[rgb(var(--cg-border-soft))] bg-[rgb(var(--cg-surface)/0.82)] px-3 py-2 text-[12px] text-[rgb(var(--cg-text-muted))]">
                            El plan no incluye componentes.
                          </div>
                        ) : (
                          <div className="space-y-3">
                            {planViewModel.components.map((component, index) => {
                              const accent = hitlComponentAccent(index);
                              return (
                                <details
                                  key={`${component.name}-${index}`}
                                  className="group rounded-xl border border-[rgb(var(--cg-border-soft))] border-l-4 bg-[rgb(var(--cg-surface-muted)/0.86)] p-3 shadow-sm transition-colors hover:bg-[rgb(var(--cg-surface)/0.94)]"
                                  style={{
                                    borderColor: `rgb(${accent} / 0.72)`,
                                    borderLeftColor: `rgb(${accent})`,
                                    borderLeftWidth: 7,
                                    backgroundImage: `linear-gradient(180deg, rgb(${accent} / 0.3), rgb(var(--cg-surface-muted)/0.86) 45%)`,
                                    boxShadow: `0 12px 24px rgb(${accent} / 0.24)`,
                                  }}
                                >
                                  <summary className="cursor-pointer list-none">
                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                      <div>
                                        <div className="flex items-center gap-2 text-[13px] font-semibold text-[rgb(var(--cg-text))]">
                                          <span
                                            className="inline-flex min-w-7 items-center justify-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold"
                                            style={{
                                              color: `rgb(${accent})`,
                                              borderColor: `rgb(${accent} / 0.9)`,
                                              backgroundColor: `rgb(${accent} / 0.35)`,
                                            }}
                                          >
                                            #{index + 1}
                                          </span>
                                          <span>{component.title}</span>
                                        </div>
                                        <div className="text-[11px] text-[rgb(var(--cg-text-muted))]">
                                          {component.name} · {component.type} · {component.inputs.length} inputs ·{" "}
                                          {component.outputs.length} outputs · {component.parameters.length} params
                                        </div>
                                      </div>
                                      <span
                                        className="inline-flex h-6 w-6 items-center justify-center rounded-full border"
                                        style={{
                                          borderColor: `rgb(${accent} / 0.82)`,
                                          backgroundColor: `rgb(${accent} / 0.35)`,
                                          color: `rgb(${accent})`,
                                        }}
                                      >
                                        <svg
                                          viewBox="0 0 20 20"
                                          fill="currentColor"
                                          className="h-4 w-4 transition-transform group-open:rotate-180"
                                          aria-hidden="true"
                                        >
                                          <path
                                            fillRule="evenodd"
                                            d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.156l3.71-3.925a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
                                            clipRule="evenodd"
                                          />
                                        </svg>
                                      </span>
                                    </div>
                                  </summary>

                                  <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                                    <div className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] px-2 py-1">
                                      <div className="text-[10px] uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Name
                                      </div>
                                      <div className="text-xs text-[rgb(var(--cg-text-soft))]">{component.name}</div>
                                    </div>
                                    <div className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] px-2 py-1">
                                      <div className="text-[10px] uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Type
                                      </div>
                                      <div className="text-xs text-[rgb(var(--cg-text-soft))]">{component.type}</div>
                                    </div>
                                  </div>

                                  <div className="mt-2 rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] px-2 py-1">
                                    <div className="text-[10px] uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                      Description
                                    </div>
                                    <p className="mt-0.5 whitespace-pre-wrap text-xs text-[rgb(var(--cg-text-soft))]">
                                      {component.description}
                                    </p>
                                  </div>

                                  <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                                    <div className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] p-2">
                                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Inputs ({component.inputs.length})
                                      </div>
                                      {component.inputs.length === 0 ? (
                                        <p className="mt-1 text-[11px] text-[rgb(var(--cg-text-muted))]">Sin inputs.</p>
                                      ) : (
                                        <div className="mt-1 space-y-1.5">
                                          {component.inputs.map((port, portIndex) => (
                                            <div
                                              key={`in-${index}-${portIndex}`}
                                              className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface-muted)/0.9)] px-2 py-1.5"
                                            >
                                              <div className="flex items-center justify-between gap-2">
                                                <span className="text-[11px] font-medium text-[rgb(var(--cg-text))]">
                                                  {port.name}
                                                </span>
                                                <span
                                                  className="rounded border px-1.5 py-0.5 text-[10px] uppercase"
                                                  style={{
                                                    borderColor: `rgb(${accent} / 0.82)`,
                                                    backgroundColor: `rgb(${accent} / 0.32)`,
                                                    color: `rgb(${accent})`,
                                                  }}
                                                >
                                                  {port.role}
                                                </span>
                                              </div>
                                              <div className="mt-1 break-all font-mono text-[11px] text-[rgb(var(--cg-text-soft))]">
                                                {port.path}
                                              </div>
                                              {port.extraFields.length > 0 ? (
                                                <div className="mt-1.5 border-t border-[rgb(var(--cg-border-soft)/0.65)] pt-1">
                                                  {port.extraFields.map((field) => (
                                                    <div
                                                      key={`in-extra-${index}-${portIndex}-${field.key}`}
                                                      className="text-[10px] text-[rgb(var(--cg-text-muted))]"
                                                    >
                                                      <span className="font-semibold">{field.key}:</span>{" "}
                                                      {field.value}
                                                    </div>
                                                  ))}
                                                </div>
                                              ) : null}
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>

                                    <div className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] p-2">
                                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Outputs ({component.outputs.length})
                                      </div>
                                      {component.outputs.length === 0 ? (
                                        <p className="mt-1 text-[11px] text-[rgb(var(--cg-text-muted))]">Sin outputs.</p>
                                      ) : (
                                        <div className="mt-1 space-y-1.5">
                                          {component.outputs.map((port, portIndex) => (
                                            <div
                                              key={`out-${index}-${portIndex}`}
                                              className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface-muted)/0.9)] px-2 py-1.5"
                                            >
                                              <div className="flex items-center justify-between gap-2">
                                                <span className="text-[11px] font-medium text-[rgb(var(--cg-text))]">
                                                  {port.name}
                                                </span>
                                                <span
                                                  className="rounded border px-1.5 py-0.5 text-[10px] uppercase"
                                                  style={{
                                                    borderColor: `rgb(${accent} / 0.82)`,
                                                    backgroundColor: `rgb(${accent} / 0.32)`,
                                                    color: `rgb(${accent})`,
                                                  }}
                                                >
                                                  {port.role}
                                                </span>
                                              </div>
                                              <div className="mt-1 break-all font-mono text-[11px] text-[rgb(var(--cg-text-soft))]">
                                                {port.path}
                                              </div>
                                              {port.extraFields.length > 0 ? (
                                                <div className="mt-1.5 border-t border-[rgb(var(--cg-border-soft)/0.65)] pt-1">
                                                  {port.extraFields.map((field) => (
                                                    <div
                                                      key={`out-extra-${index}-${portIndex}-${field.key}`}
                                                      className="text-[10px] text-[rgb(var(--cg-text-muted))]"
                                                    >
                                                      <span className="font-semibold">{field.key}:</span>{" "}
                                                      {field.value}
                                                    </div>
                                                  ))}
                                                </div>
                                              ) : null}
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  </div>

                                  <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                                    <div className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] p-2">
                                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Parameters defaults ({component.parameters.length})
                                      </div>
                                      {component.parameters.length === 0 ? (
                                        <p className="mt-1 text-[11px] text-[rgb(var(--cg-text-muted))]">Sin parámetros.</p>
                                      ) : (
                                        <div className="mt-1 space-y-1">
                                          {component.parameters.map((param) => (
                                            <div
                                              key={`param-${index}-${param.key}`}
                                              className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface-muted)/0.9)] px-2 py-1"
                                            >
                                              <div className="font-mono text-[11px] text-[rgb(var(--cg-text))]">
                                                {param.key}
                                              </div>
                                              <pre className="mt-0.5 whitespace-pre-wrap break-words font-mono text-[10px] text-[rgb(var(--cg-text-soft))]">
                                                {param.value}
                                              </pre>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>

                                    <div className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] p-2">
                                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Notes ({component.notes.length})
                                      </div>
                                      {component.notes.length === 0 ? (
                                        <p className="mt-1 text-[11px] text-[rgb(var(--cg-text-muted))]">Sin notas.</p>
                                      ) : (
                                        <ul className="mt-1 space-y-1">
                                          {component.notes.map((note, noteIndex) => (
                                            <li
                                              key={`note-${index}-${noteIndex}`}
                                              className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface-muted)/0.9)] px-2 py-1 text-[11px] text-[rgb(var(--cg-text-soft))]"
                                            >
                                              {note}
                                            </li>
                                          ))}
                                        </ul>
                                      )}
                                    </div>
                                  </div>

                                  {component.extraFields.length > 0 ? (
                                    <div className="mt-3 rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] p-2">
                                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                                        Extra fields
                                      </div>
                                      <div className="mt-1 space-y-1">
                                        {component.extraFields.map((field) => (
                                          <div
                                            key={`component-extra-${index}-${field.key}`}
                                            className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface-muted)/0.9)] px-2 py-1 text-[11px] text-[rgb(var(--cg-text-soft))]"
                                          >
                                            <span className="font-semibold text-[rgb(var(--cg-accent-strong))]">
                                              {field.key}:
                                            </span>{" "}
                                            {field.value}
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  ) : null}
                                </details>
                              );
                            })}
                          </div>
                        )}
                      </div>

                      {planViewModel.extraFields.length > 0 ? (
                        <div className="rounded border border-[rgb(var(--cg-border-soft))] bg-[rgb(var(--cg-bg)/0.58)] p-2">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                            Plan extra fields
                          </div>
                          <div className="mt-1 space-y-1">
                            {planViewModel.extraFields.map((field) => (
                              <div
                                key={`plan-extra-${field.key}`}
                                className="rounded border border-[rgb(var(--cg-border-soft)/0.85)] bg-[rgb(var(--cg-surface)/0.82)] px-2 py-1 text-[11px] text-[rgb(var(--cg-text-soft))]"
                              >
                                <span className="font-semibold text-[rgb(var(--cg-accent-strong))]">
                                  {field.key}:
                                </span>{" "}
                                {field.value}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>

                    <label className="mt-4 block">
                      <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[rgb(var(--cg-accent-strong))]">
                        Feedback para revisar plan
                      </span>
                      <textarea
                        value={feedback}
                        onChange={(event) => setFeedback(event.target.value)}
                        rows={5}
                        className={`${FORM_FIELD_CLASS} border-[rgb(var(--cg-border-soft))] bg-[rgb(var(--cg-surface)/0.92)] text-[rgb(var(--cg-text-soft))]`}
                        placeholder="Qué debería cambiar el agente..."
                      />
                    </label>

                    <div className="mt-3 flex gap-2">
                      <button
                        type="button"
                        onClick={() => void submitDecision(true)}
                        disabled={decisionLoading}
                        className="cg-btn cg-btn-success inline-flex items-center justify-center rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-emerald-900/50"
                      >
                        Aprobar plan
                      </button>
                      <button
                        type="button"
                        onClick={() => void submitDecision(false)}
                        disabled={decisionLoading}
                        className="cg-btn cg-btn-warning inline-flex items-center justify-center rounded-md bg-amber-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-amber-900/60 disabled:text-amber-200"
                      >
                        Solicitar cambios
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
                    <h3 className="cg-card-title text-xs font-semibold uppercase tracking-wide text-slate-300">
                      Consola de ejecución
                    </h3>
                    <div className="flex flex-wrap items-center gap-2 text-[11px]">
                      <select
                        value={logFilter}
                        onChange={(event) => setLogFilter(event.target.value as LogFilter)}
                        className="cg-input rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-200"
                      >
                        <option value="ALL">all</option>
                        <option value="ERROR">error</option>
                        <option value="WARNING">warning</option>
                        <option value="INFO">info</option>
                        <option value="DEBUG">debug</option>
                        <option value="STDOUT">stdout</option>
                        <option value="STDERR">stderr</option>
                      </select>
                      <button
                        type="button"
                        onClick={() => setAutoScrollLogs((prev) => !prev)}
                        className="cg-btn cg-btn-muted rounded border border-slate-700 px-2 py-1 text-slate-200 hover:bg-slate-800"
                      >
                        {autoScrollLogs ? "autoscroll on" : "autoscroll off"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setLogEntries([])}
                        className="cg-btn cg-btn-muted rounded border border-slate-700 px-2 py-1 text-slate-200 hover:bg-slate-800"
                      >
                        clear
                      </button>
                    </div>
                  </div>
                  <div
                    ref={logContainerRef}
                    className="cg-terminal min-h-0 flex-1 overflow-y-auto p-2 font-mono text-[11px] whitespace-pre-wrap break-words"
                  >
                    {filteredLogs.length === 0 ? (
                      <p className="cg-muted">Sin logs todavía.</p>
                    ) : (
                      filteredLogs.map((entry) => (
                        <div key={entry.id} className={`leading-5 ${levelClass(entry.level)}`}>
                          <span className="cg-muted">[{entry.level}]</span>{" "}
                          <span className="cg-muted">{entry.source}:</span>{" "}
                          {entry.line}
                        </div>
                      ))
                    )}
                  </div>
                </>
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
