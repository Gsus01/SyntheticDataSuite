"use client";

import Link from "next/link";
import React from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  MarkerType,
  MiniMap,
  type Node,
  type NodeProps,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";

import {
  cancelComponentGenerationRun,
  createComponentGenerationRun,
  fetchComponentGenerationRun,
  type ComponentGenerationRunEvent,
  type ComponentGenerationRunSnapshot,
  subscribeComponentGenerationEvents,
  submitComponentGenerationDecision,
} from "@/lib/component-generation";

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
  if (status === "queued") return "bg-slate-700 text-slate-100";
  if (status === "running") return "bg-cyan-600 text-white";
  if (status === "waiting_decision") return "bg-amber-500 text-slate-950";
  if (status === "succeeded") return "bg-emerald-600 text-white";
  if (status === "failed") return "bg-rose-600 text-white";
  if (status === "canceled") return "bg-amber-600 text-white";
  return "bg-slate-700 text-slate-100";
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
  if (normalized === "ERROR" || normalized === "CRITICAL") return "text-rose-300";
  if (normalized === "WARNING" || normalized === "WARN") return "text-amber-300";
  if (normalized === "DEBUG") return "text-sky-300";
  if (normalized === "STDERR") return "text-rose-200";
  if (normalized === "STDOUT") return "text-emerald-200";
  return "text-slate-200";
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
  if (sourceState === "failed" || targetState === "failed") return "#fb7185";
  if (sourceState === "canceled" || targetState === "canceled") return "#f59e0b";
  if (targetState === "running" || sourceState === "running") return "#22d3ee";
  if (sourceState === "completed" && targetState === "completed") return "#34d399";
  if (targetState === "waiting_decision") return "#f59e0b";
  return "#475569";
}

function nodeTheme(state: WorkflowNodeState): {
  border: string;
  bg: string;
  text: string;
  pulseClass: string;
} {
  if (state === "running") {
    return {
      border: "border-cyan-400/80",
      bg: "bg-gradient-to-br from-cyan-500/20 to-cyan-700/20",
      text: "text-cyan-100",
      pulseClass: "node-animate-running",
    };
  }
  if (state === "waiting_decision") {
    return {
      border: "border-amber-400/80",
      bg: "bg-gradient-to-br from-amber-500/20 to-amber-700/20",
      text: "text-amber-100",
      pulseClass: "node-animate-canceled",
    };
  }
  if (state === "completed") {
    return {
      border: "border-emerald-400/80",
      bg: "bg-gradient-to-br from-emerald-500/18 to-emerald-700/18",
      text: "text-emerald-100",
      pulseClass: "node-animate-completed",
    };
  }
  if (state === "failed") {
    return {
      border: "border-rose-400/90",
      bg: "bg-gradient-to-br from-rose-500/22 to-rose-700/18",
      text: "text-rose-100",
      pulseClass: "node-animate-failed",
    };
  }
  if (state === "canceled") {
    return {
      border: "border-amber-500/80",
      bg: "bg-gradient-to-br from-amber-500/20 to-amber-800/20",
      text: "text-amber-100",
      pulseClass: "node-animate-canceled",
    };
  }
  return {
    border: "border-slate-600",
    bg: "bg-slate-800/65",
    text: "text-slate-200",
    pulseClass: "node-animate-pending",
  };
}

function stateLabel(state: WorkflowNodeState): string {
  if (state === "waiting_decision") return "waiting approval";
  return state;
}

function WorkflowGraphNode({ data }: NodeProps<WorkflowGraphNodeData>) {
  const theme = nodeTheme(data.state);
  return (
    <div
      className={`min-w-[170px] rounded-xl border-2 px-3 py-2 shadow-lg backdrop-blur-sm ${theme.border} ${theme.bg} ${theme.text} ${theme.pulseClass}`}
    >
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={false}
        className="!h-2 !w-2 !border !border-slate-700 !bg-slate-300/80"
      />
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={false}
        className="!h-2 !w-2 !border !border-slate-700 !bg-slate-300/80"
      />
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider">{data.label}</span>
        <span className="rounded-full border border-white/20 px-1.5 py-0.5 text-[9px] uppercase">
          {stateLabel(data.state)}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-slate-300/90">
        {data.message && data.message.trim()
          ? data.message
          : "LangGraph step"}
      </div>
    </div>
  );
}

const graphNodeTypes = {
  workflowNode: WorkflowGraphNode,
};

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
  const [streamConnected, setStreamConnected] = React.useState(false);
  const [logFilter, setLogFilter] = React.useState<LogFilter>("ALL");
  const [autoScrollLogs, setAutoScrollLogs] = React.useState(true);
  const [rememberOpenrouterKey, setRememberOpenrouterKey] = React.useState(false);
  const [formStorageReady, setFormStorageReady] = React.useState(false);

  const eventSourceRef = React.useRef<EventSource | null>(null);
  const logContainerRef = React.useRef<HTMLDivElement | null>(null);
  const nextLogIdRef = React.useRef(1);

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

  const graphNodes = React.useMemo<Node<WorkflowGraphNodeData>[]>(
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
          fill: "#cbd5e1",
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

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="flex h-16 items-center justify-between border-b border-slate-800 bg-slate-950 px-5">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:bg-slate-800"
          >
            Volver al editor
          </Link>
          <div>
            <h1 className="text-sm font-semibold">
              Generación Automática de Componentes
            </h1>
            <p className="text-[11px] text-slate-400">
              Logs estilo CLI, grafo en vivo y revisión HITL completa.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {run ? (
            <span
              className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide ${runStatusBadge(
                run.status
              )}`}
            >
              {run.status.replace("_", " ")}
            </span>
          ) : (
            <span className="text-xs text-slate-400">Sin ejecución activa</span>
          )}
          <span
            className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ${
              streamConnected
                ? "bg-emerald-600 text-white"
                : "bg-slate-700 text-slate-200"
            }`}
          >
            {streamConnected ? "SSE online" : "SSE offline"}
          </span>
        </div>
      </header>

      <div className="grid h-[calc(100vh-4rem)] grid-cols-1 lg:grid-cols-[360px_1fr]">
        <aside className="flex min-h-0 flex-col overflow-y-auto border-r border-slate-800 bg-slate-900 p-4">
          <div className="space-y-4">
            <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
                Archivos de entrada
              </h2>
              <label className="mt-3 flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-slate-700 bg-slate-950/60 px-3 py-4 text-center text-xs text-slate-300 hover:border-slate-600">
                <span className="font-medium">Seleccionar archivos</span>
                <span className="mt-1 text-[11px] text-slate-500">
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
                  <p className="text-[11px] text-slate-500">Sin archivos seleccionados.</p>
                ) : (
                  selectedFiles.map((file) => (
                    <div
                      key={`${file.name}-${file.size}`}
                      className="rounded border border-slate-800 bg-slate-950/60 px-2 py-1.5 text-[11px]"
                    >
                      <div className="truncate font-medium text-slate-200">{file.name}</div>
                      <div className="text-slate-500">{formatBytes(file.size)}</div>
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
                Configuración
              </h2>
              <div className="mt-3 space-y-3 text-xs">
                <label className="flex items-center justify-between gap-3">
                  <span>Incluir markdown</span>
                  <input
                    type="checkbox"
                    checked={form.includeMarkdown}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        includeMarkdown: event.target.checked,
                      }))
                    }
                  />
                </label>

                <label className="block">
                  <span className="mb-1 block text-[11px] text-slate-400">Max chars por fichero</span>
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
                    className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                  />
                </label>

                <label className="block">
                  <span className="mb-1 block text-[11px] text-slate-400">Provider</span>
                  <select
                    value={form.provider}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        provider: event.target.value as "ollama" | "openrouter",
                      }))
                    }
                    className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                  >
                    <option value="ollama">ollama</option>
                    <option value="openrouter">openrouter</option>
                  </select>
                </label>

                <label className="block">
                  <span className="mb-1 block text-[11px] text-slate-400">Modelo</span>
                  <input
                    type="text"
                    value={form.model}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, model: event.target.value }))
                    }
                    className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                  />
                </label>

                <label className="block">
                  <span className="mb-1 block text-[11px] text-slate-400">Temperatura</span>
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
                    className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                  />
                </label>

                <label className="flex items-center justify-between gap-3">
                  <span>Desactivar LLM</span>
                  <input
                    type="checkbox"
                    checked={form.noLlm}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, noLlm: event.target.checked }))
                    }
                  />
                </label>

                <label className="flex items-center justify-between gap-3">
                  <span>Sin structured output</span>
                  <input
                    type="checkbox"
                    checked={form.noStructuredOutput}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        noStructuredOutput: event.target.checked,
                      }))
                    }
                  />
                </label>

                {form.provider === "ollama" ? (
                  <label className="block">
                    <span className="mb-1 block text-[11px] text-slate-400">Ollama URL</span>
                    <input
                      type="text"
                      value={form.ollamaUrl}
                      onChange={(event) =>
                        setForm((prev) => ({ ...prev, ollamaUrl: event.target.value }))
                      }
                      className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                    />
                  </label>
                ) : (
                  <>
                    <label className="block">
                      <span className="mb-1 block text-[11px] text-slate-400">OpenRouter URL</span>
                      <input
                        type="text"
                        value={form.openrouterUrl}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterUrl: event.target.value,
                          }))
                        }
                        className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-[11px] text-slate-400">OpenRouter Key</span>
                      <input
                        type="password"
                        value={form.openrouterKey}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterKey: event.target.value,
                          }))
                        }
                        className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                      />
                    </label>
                    <label className="flex items-center justify-between gap-3 rounded border border-slate-800 bg-slate-950/60 px-2 py-1.5">
                      <span className="text-[11px] text-slate-300">
                        Recordar API key en este navegador
                      </span>
                      <input
                        type="checkbox"
                        checked={rememberOpenrouterKey}
                        onChange={(event) =>
                          setRememberOpenrouterKey(event.target.checked)
                        }
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1 block text-[11px] text-slate-400">Provider order</span>
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
                        className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-100"
                      />
                    </label>
                  </>
                )}

                <button
                  type="button"
                  onClick={resetLocalSettings}
                  className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-[11px] font-medium text-slate-200 hover:bg-slate-800"
                >
                  Restablecer ajustes locales
                </button>
              </div>
            </section>

            <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={startRun}
                  disabled={starting || (run ? !isTerminal(run.status) : false)}
                  className="inline-flex flex-1 items-center justify-center rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-900/60"
                >
                  {starting ? "Iniciando..." : "Iniciar generación"}
                </button>
                <button
                  type="button"
                  onClick={cancelRun}
                  disabled={!run || !run.canCancel || cancelLoading}
                  className="inline-flex flex-1 items-center justify-center rounded-md bg-rose-600 px-3 py-2 text-xs font-semibold text-white hover:bg-rose-500 disabled:cursor-not-allowed disabled:bg-rose-900/60"
                >
                  {cancelLoading ? "Interrumpiendo..." : "Interrumpir"}
                </button>
              </div>
              {error ? (
                <p className="mt-2 text-[11px] text-rose-300">{error}</p>
              ) : null}
            </section>
          </div>
        </aside>

        <main className="flex min-h-0 flex-col">
          <section className="h-[44%] border-b border-slate-800 px-4 py-3">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
              LangGraph en tiempo real
            </h2>
            <div className="h-[calc(100%-1.8rem)] rounded-xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950">
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
                <Background color="#273244" gap={28} size={1} />
                <MiniMap
                  pannable
                  zoomable
                  className="!bg-slate-950/70 !border !border-slate-700"
                  maskColor="rgba(2,6,23,0.6)"
                />
                <Controls />
              </ReactFlow>
            </div>
          </section>

          <section className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden p-4 xl:grid-cols-[0.95fr_1.45fr]">
            <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
              {run?.awaitingDecision ? (
                <div className="rounded-xl border border-amber-500/35 bg-amber-500/10 p-3">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-amber-100">
                      Revisión HITL
                    </h3>
                    <p className="mt-1 text-[11px] text-amber-100/70">
                      Se muestra el plan completo por campos para revisar cada detalle antes de aprobar.
                    </p>
                  </div>

                  <div className="mt-3 space-y-3">
                    <div className="rounded border border-amber-400/20 bg-black/30 p-2">
                      <div className="text-[11px] font-semibold uppercase tracking-wide text-amber-100/80">
                        Rationale
                      </div>
                      <p className="mt-1 whitespace-pre-wrap text-xs text-amber-50">
                        {planViewModel.rationale}
                      </p>
                    </div>

                    <div className="rounded border border-amber-400/20 bg-black/30 p-2">
                      <div className="text-[11px] font-semibold uppercase tracking-wide text-amber-100/80">
                        Assumptions
                      </div>
                      {planViewModel.assumptions.length === 0 ? (
                        <p className="mt-1 text-xs text-amber-100/70">
                          Sin assumptions.
                        </p>
                      ) : (
                        <ul className="mt-1 space-y-1 text-xs text-amber-50">
                          {planViewModel.assumptions.map((assumption, index) => (
                            <li key={`assumption-${index}`} className="rounded border border-amber-300/15 bg-black/20 px-2 py-1">
                              {assumption}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>

                    <div>
                      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-amber-100/80">
                        Components ({planViewModel.components.length})
                      </div>
                      {planViewModel.components.length === 0 ? (
                        <div className="rounded border border-amber-400/20 bg-black/30 px-2 py-1.5 text-xs text-amber-100/70">
                          El plan no incluye componentes.
                        </div>
                      ) : (
                        <div className="max-h-[28rem] space-y-3 overflow-y-auto pr-1">
                          {planViewModel.components.map((component, index) => (
                            <details
                              key={`${component.name}-${index}`}
                              className="group rounded border border-amber-400/25 bg-black/30 p-3"
                            >
                              <summary className="cursor-pointer list-none">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <div>
                                    <div className="text-xs font-semibold text-amber-50">
                                      #{index + 1} {component.title}
                                    </div>
                                    <div className="text-[11px] text-amber-100/75">
                                      {component.name} · {component.type} · {component.inputs.length} inputs ·{" "}
                                      {component.outputs.length} outputs · {component.parameters.length} params
                                    </div>
                                  </div>
                                  <span className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-amber-300/30 text-amber-100/90">
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
                                <div className="rounded border border-amber-300/15 bg-black/25 px-2 py-1">
                                  <div className="text-[10px] uppercase tracking-wide text-amber-100/70">
                                    Name
                                  </div>
                                  <div className="text-xs text-amber-50">{component.name}</div>
                                </div>
                                <div className="rounded border border-amber-300/15 bg-black/25 px-2 py-1">
                                  <div className="text-[10px] uppercase tracking-wide text-amber-100/70">
                                    Type
                                  </div>
                                  <div className="text-xs text-amber-50">{component.type}</div>
                                </div>
                              </div>

                              <div className="mt-2 rounded border border-amber-300/15 bg-black/25 px-2 py-1">
                                <div className="text-[10px] uppercase tracking-wide text-amber-100/70">
                                  Description
                                </div>
                                <p className="mt-0.5 whitespace-pre-wrap text-xs text-amber-50">
                                  {component.description}
                                </p>
                              </div>

                              <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
                                <div className="rounded border border-amber-300/15 bg-black/25 p-2">
                                  <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-100/70">
                                    Inputs ({component.inputs.length})
                                  </div>
                                  {component.inputs.length === 0 ? (
                                    <p className="mt-1 text-[11px] text-amber-100/60">Sin inputs.</p>
                                  ) : (
                                    <div className="mt-1 space-y-1.5">
                                      {component.inputs.map((port, portIndex) => (
                                        <div
                                          key={`in-${index}-${portIndex}`}
                                          className="rounded border border-amber-300/15 bg-black/30 px-2 py-1.5"
                                        >
                                          <div className="flex items-center justify-between gap-2">
                                            <span className="text-[11px] font-medium text-amber-50">
                                              {port.name}
                                            </span>
                                            <span className="rounded border border-amber-200/20 px-1.5 py-0.5 text-[10px] uppercase text-amber-100/80">
                                              {port.role}
                                            </span>
                                          </div>
                                          <div className="mt-1 break-all font-mono text-[11px] text-amber-100/90">
                                            {port.path}
                                          </div>
                                          {port.extraFields.length > 0 ? (
                                            <div className="mt-1.5 border-t border-amber-300/10 pt-1">
                                              {port.extraFields.map((field) => (
                                                <div key={`in-extra-${index}-${portIndex}-${field.key}`} className="text-[10px] text-amber-100/75">
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

                                <div className="rounded border border-amber-300/15 bg-black/25 p-2">
                                  <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-100/70">
                                    Outputs ({component.outputs.length})
                                  </div>
                                  {component.outputs.length === 0 ? (
                                    <p className="mt-1 text-[11px] text-amber-100/60">Sin outputs.</p>
                                  ) : (
                                    <div className="mt-1 space-y-1.5">
                                      {component.outputs.map((port, portIndex) => (
                                        <div
                                          key={`out-${index}-${portIndex}`}
                                          className="rounded border border-amber-300/15 bg-black/30 px-2 py-1.5"
                                        >
                                          <div className="flex items-center justify-between gap-2">
                                            <span className="text-[11px] font-medium text-amber-50">
                                              {port.name}
                                            </span>
                                            <span className="rounded border border-amber-200/20 px-1.5 py-0.5 text-[10px] uppercase text-amber-100/80">
                                              {port.role}
                                            </span>
                                          </div>
                                          <div className="mt-1 break-all font-mono text-[11px] text-amber-100/90">
                                            {port.path}
                                          </div>
                                          {port.extraFields.length > 0 ? (
                                            <div className="mt-1.5 border-t border-amber-300/10 pt-1">
                                              {port.extraFields.map((field) => (
                                                <div key={`out-extra-${index}-${portIndex}-${field.key}`} className="text-[10px] text-amber-100/75">
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
                                <div className="rounded border border-amber-300/15 bg-black/25 p-2">
                                  <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-100/70">
                                    Parameters defaults ({component.parameters.length})
                                  </div>
                                  {component.parameters.length === 0 ? (
                                    <p className="mt-1 text-[11px] text-amber-100/60">Sin parámetros.</p>
                                  ) : (
                                    <div className="mt-1 space-y-1">
                                      {component.parameters.map((param) => (
                                        <div key={`param-${index}-${param.key}`} className="rounded border border-amber-300/15 bg-black/30 px-2 py-1">
                                          <div className="font-mono text-[11px] text-amber-100">
                                            {param.key}
                                          </div>
                                          <pre className="mt-0.5 whitespace-pre-wrap break-words font-mono text-[10px] text-amber-50">
                                            {param.value}
                                          </pre>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>

                                <div className="rounded border border-amber-300/15 bg-black/25 p-2">
                                  <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-100/70">
                                    Notes ({component.notes.length})
                                  </div>
                                  {component.notes.length === 0 ? (
                                    <p className="mt-1 text-[11px] text-amber-100/60">Sin notas.</p>
                                  ) : (
                                    <ul className="mt-1 space-y-1">
                                      {component.notes.map((note, noteIndex) => (
                                        <li
                                          key={`note-${index}-${noteIndex}`}
                                          className="rounded border border-amber-300/15 bg-black/30 px-2 py-1 text-[11px] text-amber-50"
                                        >
                                          {note}
                                        </li>
                                      ))}
                                    </ul>
                                  )}
                                </div>
                              </div>

                              {component.extraFields.length > 0 ? (
                                <div className="mt-3 rounded border border-amber-300/15 bg-black/25 p-2">
                                  <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-100/70">
                                    Extra fields
                                  </div>
                                  <div className="mt-1 space-y-1">
                                    {component.extraFields.map((field) => (
                                      <div key={`component-extra-${index}-${field.key}`} className="rounded border border-amber-300/15 bg-black/30 px-2 py-1 text-[11px] text-amber-50">
                                        <span className="font-semibold text-amber-100">{field.key}:</span>{" "}
                                        {field.value}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                            </details>
                          ))}
                        </div>
                      )}
                    </div>

                    {planViewModel.extraFields.length > 0 ? (
                      <div className="rounded border border-amber-400/20 bg-black/30 p-2">
                        <div className="text-[11px] font-semibold uppercase tracking-wide text-amber-100/80">
                          Plan extra fields
                        </div>
                        <div className="mt-1 space-y-1">
                          {planViewModel.extraFields.map((field) => (
                            <div key={`plan-extra-${field.key}`} className="rounded border border-amber-300/15 bg-black/25 px-2 py-1 text-[11px] text-amber-50">
                              <span className="font-semibold text-amber-100">{field.key}:</span>{" "}
                              {field.value}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}

                  </div>

                  <label className="mt-3 block">
                    <span className="mb-1 block text-[11px] text-amber-100/80">
                      Feedback para revisar plan
                    </span>
                    <textarea
                      value={feedback}
                      onChange={(event) => setFeedback(event.target.value)}
                      rows={4}
                      className="w-full rounded border border-amber-300/25 bg-black/25 px-2 py-1.5 text-xs text-amber-50"
                      placeholder="Qué debería cambiar el agente..."
                    />
                  </label>

                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      onClick={() => void submitDecision(true)}
                      disabled={decisionLoading}
                      className="inline-flex items-center justify-center rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-emerald-900/50"
                    >
                      Aprobar plan
                    </button>
                    <button
                      type="button"
                      onClick={() => void submitDecision(false)}
                      disabled={decisionLoading}
                      className="inline-flex items-center justify-center rounded-md bg-amber-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-amber-900/60 disabled:text-amber-200"
                    >
                      Solicitar cambios
                    </button>
                  </div>
                </div>
              ) : null}

              <div className="rounded-xl border border-slate-800 bg-slate-900 p-3">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
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
                      <span className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase ${runStatusBadge(run.status)}`}>
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
                  <p className="mt-2 text-xs text-slate-500">Aún no has lanzado ninguna ejecución.</p>
                )}
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900 p-3">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
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
                    <div className="max-h-28 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-slate-300">
                      {(run.reviewReport || "(sin reporte)").trim() || "(sin reporte)"}
                    </div>
                    <div className="max-h-28 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-slate-400">
                      {events.length === 0
                        ? "Sin eventos todavía."
                        : events
                            .slice(-20)
                            .map((event) => {
                              const payload = event.payload || {};
                              const main =
                                event.type === "log_line" && payload.line
                                  ? String(payload.line)
                                  : event.type;
                              return `#${event.seq} ${formatTimestamp(event.timestamp)} ${main}`;
                            })
                            .join("\n")}
                    </div>
                    <div className="max-h-20 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-rose-300">
                      {run.error || "(sin error)"}
                    </div>
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-slate-500">Sin artefactos todavía.</p>
                )}
              </div>
            </div>

            <div className="flex min-h-0 flex-col rounded-xl border border-slate-800 bg-slate-900">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
                  Consola de ejecución
                </h3>
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <select
                    value={logFilter}
                    onChange={(event) => setLogFilter(event.target.value as LogFilter)}
                    className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-200"
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
                    className="rounded border border-slate-700 px-2 py-1 text-slate-200 hover:bg-slate-800"
                  >
                    {autoScrollLogs ? "autoscroll on" : "autoscroll off"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setLogEntries([])}
                    className="rounded border border-slate-700 px-2 py-1 text-slate-200 hover:bg-slate-800"
                  >
                    clear
                  </button>
                </div>
              </div>
              <div
                ref={logContainerRef}
                className="min-h-0 flex-1 overflow-y-auto p-2 font-mono text-[11px] whitespace-pre-wrap break-words"
              >
                {filteredLogs.length === 0 ? (
                  <p className="text-slate-500">Sin logs todavía.</p>
                ) : (
                  filteredLogs.map((entry) => (
                    <div key={entry.id} className={`leading-5 ${levelClass(entry.level)}`}>
                      <span className="text-slate-500">[{entry.level}]</span>{" "}
                      <span className="text-slate-400">{entry.source}:</span>{" "}
                      {entry.line}
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
