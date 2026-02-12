import type {
  ComponentGenerationRunEvent,
  ComponentGenerationRunSnapshot,
} from "@/lib/component-generation";

import { DEFAULT_FORM, TERMINAL_STATUSES } from "./constants";
import type {
  FormState,
  IntegrationSummaryComponent,
  IntegrationSummaryFile,
  IntegrationSummaryView,
  KeyValueLine,
  LogEntry,
  LogFilter,
  NodeTheme,
  PlanComponentView,
  PlanPortView,
  PlanViewModel,
  WorkflowNodeState,
} from "./types";
import type { BadgeVariant } from "./ui/primitives";

export function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
}

export function runStatusBadge(status: string): BadgeVariant {
  if (status === "running") return "running";
  if (status === "waiting_decision") return "waiting";
  if (status === "succeeded") return "success";
  if (status === "failed") return "danger";
  if (status === "canceled") return "waiting";
  return "neutral";
}

export function eventSummary(event: ComponentGenerationRunEvent): string {
  const payload = event.payload || {};
  const stage =
    typeof payload.stage === "string" && payload.stage.trim()
      ? payload.stage.trim()
      : "plan";
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
      return `Awaiting ${stage} approval`;
    case "decision_submitted":
      return `Decision submitted (${stage}): ${payload.approved ? "Approved" : "Rejected"}`;
    case "resumed":
      return "Execution resumed after decision";
    case "integration_summary_ready":
      return "Integration summary ready for confirmation";
    case "integration_build_started":
      return "Docker build started";
    case "integration_build_component_started":
      return `Building image for ${payload.component || "component"}`;
    case "integration_build_component_succeeded":
      return `Image built for ${payload.component || "component"}`;
    case "integration_registration_started":
      return "Component registration started";
    case "integration_registration_component_succeeded":
      return `Component registered: ${payload.component || "component"}`;
    case "integration_skipped":
      return "Integration skipped by user";
    case "integration_failed":
      return `Integration failed: ${payload.error || "Unknown error"}`;
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

export function parseSnapshotLogLine(rawLine: string): Omit<LogEntry, "id"> {
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

export function levelClass(level: string): string {
  const normalized = level.toUpperCase();
  if (normalized === "ERROR" || normalized === "CRITICAL") {
    return "text-red-700 dark:text-red-300";
  }
  if (normalized === "WARNING" || normalized === "WARN") {
    return "text-amber-700 dark:text-amber-300";
  }
  if (normalized === "DEBUG") return "text-sky-700 dark:text-sky-300";
  if (normalized === "STDERR") return "text-red-700 dark:text-red-300";
  if (normalized === "STDOUT") return "text-emerald-700 dark:text-emerald-300";
  return "text-slate-700 dark:text-slate-300";
}

export function shouldIncludeLog(entry: LogEntry, filter: LogFilter): boolean {
  if (filter === "ALL") return true;
  const level = entry.level.toUpperCase();
  if (filter === "WARNING") return level === "WARNING" || level === "WARN";
  return level === filter;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

export function parsePersistedFormState(
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
    typeof maxCharsRaw === "number" && Number.isFinite(maxCharsRaw) && maxCharsRaw > 0
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

export function parsePlanForDisplay(
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

function parseIntegrationFile(rawFile: unknown): IntegrationSummaryFile | null {
  const file = asRecord(rawFile);
  if (!file) return null;
  const name =
    typeof file.name === "string" && file.name.trim() ? file.name.trim() : "(sin nombre)";
  const path =
    typeof file.path === "string" && file.path.trim() ? file.path.trim() : "(sin path)";
  return { name, path };
}

function parseIntegrationComponent(rawComponent: unknown): IntegrationSummaryComponent | null {
  const component = asRecord(rawComponent);
  if (!component) return null;

  const filesRaw = Array.isArray(component.files) ? component.files : [];
  const files = filesRaw
    .map(parseIntegrationFile)
    .filter((item): item is IntegrationSummaryFile => Boolean(item));

  return {
    name:
      typeof component.name === "string" && component.name.trim()
        ? component.name
        : "(sin nombre)",
    title:
      typeof component.title === "string" && component.title.trim()
        ? component.title
        : "(sin título)",
    version:
      typeof component.version === "string" && component.version.trim()
        ? component.version
        : "(sin versión)",
    type:
      typeof component.type === "string" && component.type.trim()
        ? component.type
        : "(sin tipo)",
    image:
      typeof component.image === "string" && component.image.trim()
        ? component.image
        : "(sin imagen)",
    description:
      typeof component.description === "string" && component.description.trim()
        ? component.description
        : "",
    files,
  };
}

export function parseIntegrationSummaryForDisplay(
  pendingIntegration: Record<string, unknown> | null | undefined
): IntegrationSummaryView {
  const summary = asRecord(pendingIntegration) || {};
  const componentsRaw = Array.isArray(summary.components) ? summary.components : [];
  const components = componentsRaw
    .map(parseIntegrationComponent)
    .filter((item): item is IntegrationSummaryComponent => Boolean(item));

  const countRaw = summary.componentCount;
  const componentCount =
    typeof countRaw === "number" && Number.isFinite(countRaw) && countRaw >= 0
      ? Math.trunc(countRaw)
      : components.length;

  return {
    componentCount,
    components,
  };
}

export function workflowNodeStateFor(
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

export function edgeColorFor(
  sourceState: WorkflowNodeState,
  targetState: WorkflowNodeState
): string {
  if (sourceState === "failed" || targetState === "failed") return "#ef4444";
  if (sourceState === "canceled" || targetState === "canceled") return "#f59e0b";
  if (targetState === "running" || sourceState === "running") return "#0ea5e9";
  if (sourceState === "completed" && targetState === "completed") {
    return "#22c55e";
  }
  if (targetState === "waiting_decision") return "#f59e0b";
  return "#64748b";
}

export function nodeTheme(state: WorkflowNodeState): NodeTheme {
  if (state === "running") {
    return {
      toneClass:
        "border-sky-400 bg-sky-50 text-sky-800 dark:border-sky-700 dark:bg-sky-900/30 dark:text-sky-200",
      pulseClass: "animate-pulse",
    };
  }
  if (state === "waiting_decision") {
    return {
      toneClass:
        "border-amber-400 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-200",
      pulseClass: "animate-pulse",
    };
  }
  if (state === "completed") {
    return {
      toneClass:
        "border-emerald-400 bg-emerald-50 text-emerald-800 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200",
      pulseClass: "",
    };
  }
  if (state === "failed") {
    return {
      toneClass:
        "border-red-400 bg-red-50 text-red-800 dark:border-red-700 dark:bg-red-900/30 dark:text-red-200",
      pulseClass: "animate-pulse",
    };
  }
  if (state === "canceled") {
    return {
      toneClass:
        "border-amber-400 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-200",
      pulseClass: "animate-pulse",
    };
  }
  return {
    toneClass:
      "border-slate-300 bg-slate-50 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200",
    pulseClass: "",
  };
}

export function stateLabel(state: WorkflowNodeState): string {
  if (state === "waiting_decision") return "waiting approval";
  return state;
}

export function updateRunFromEvent(
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
    next.decisionStage = null;
  } else if (event.type === "plan_proposed" || event.type === "waiting_decision") {
    const stage =
      typeof payload.stage === "string" && payload.stage.trim()
        ? payload.stage.trim()
        : "plan";
    if (event.type === "waiting_decision") {
      next.status = "waiting_decision";
      next.decisionStage = stage === "integration" ? "integration" : "plan";
      if (next.decisionStage === "integration") {
        const summary = payload.summary;
        if (summary && typeof summary === "object") {
          next.pendingIntegration = summary as Record<string, unknown>;
        }
        next.integrationStatus = "waiting_confirmation";
      }
    }
    if (stage !== "integration") {
      const plan = payload.plan;
      if (plan && typeof plan === "object") {
        next.pendingPlan = plan as Record<string, unknown>;
      }
      if (typeof payload.prettyPlan === "string") {
        next.pendingPrettyPlan = payload.prettyPlan;
      }
    }
  } else if (event.type === "resumed") {
    next.status = "running";
    next.decisionStage = null;
  } else if (event.type === "integration_summary_ready") {
    const summary = payload.summary;
    if (summary && typeof summary === "object") {
      next.pendingIntegration = summary as Record<string, unknown>;
    }
  } else if (
    event.type === "integration_build_started" ||
    event.type === "integration_build_component_started" ||
    event.type === "integration_build_component_succeeded"
  ) {
    next.integrationStatus = "building";
  } else if (
    event.type === "integration_registration_started" ||
    event.type === "integration_registration_component_succeeded"
  ) {
    next.integrationStatus = "registering";
    const result = payload.result;
    if (result && typeof result === "object") {
      next.integrationResult = result as Record<string, unknown>;
    }
  } else if (event.type === "integration_skipped") {
    next.integrationStatus = "skipped_by_user";
    next.decisionStage = null;
    const result = payload.result;
    if (result && typeof result === "object") {
      next.integrationResult = result as Record<string, unknown>;
    }
  } else if (event.type === "integration_failed") {
    next.integrationStatus = "failed";
    next.decisionStage = null;
    if (typeof payload.error === "string") {
      next.error = payload.error;
    }
  } else if (event.type === "run_finished") {
    next.status = "succeeded";
    next.decisionStage = null;
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
    if (typeof payload.integrationStatus === "string") {
      next.integrationStatus = payload.integrationStatus;
    }
    const integrationResult = payload.integrationResult;
    if (integrationResult && typeof integrationResult === "object") {
      next.integrationResult = integrationResult as Record<string, unknown>;
    }
  } else if (event.type === "run_failed") {
    next.status = "failed";
    next.decisionStage = null;
    if (typeof payload.error === "string") {
      next.error = payload.error;
    }
  } else if (event.type === "run_canceled") {
    next.status = "canceled";
    next.decisionStage = null;
    next.error = "Run canceled by user.";
  } else if (
    event.type === "node_started" ||
    event.type === "node_completed" ||
    event.type === "node_failed"
  ) {
    const nodeName =
      typeof payload.node === "string" && payload.node.trim() ? payload.node.trim() : "";
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
