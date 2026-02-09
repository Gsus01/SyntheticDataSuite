import { API_BASE, buildApiUrl } from "@/lib/api";

export type ComponentGenerationRunStatus =
  | "queued"
  | "running"
  | "waiting_decision"
  | "succeeded"
  | "failed"
  | "canceled";

export type ComponentGenerationNodeState = {
  state: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  message?: string | null;
};

export type ComponentGenerationInputFile = {
  filename: string;
  size: number;
  truncated?: boolean;
  contentType?: string | null;
  storedPath?: string;
};

export type ComponentGenerationRunSnapshot = {
  runId: string;
  status: ComponentGenerationRunStatus;
  createdAt: string;
  updatedAt: string;
  sessionDir: string;
  inputFiles: ComponentGenerationInputFile[];
  options: Record<string, unknown>;
  nodeStates: Record<string, ComponentGenerationNodeState>;
  pendingPlan?: Record<string, unknown> | null;
  generatedIndex: Record<string, Record<string, string>>;
  reviewReport?: string | null;
  reviewStatus?: string | null;
  integrationReport?: string | null;
  error?: string | null;
  canCancel: boolean;
  awaitingDecision: boolean;
  lastSeq: number;
};

export type ComponentGenerationRunEvent = {
  seq: number;
  timestamp: string;
  type: string;
  payload: Record<string, unknown>;
};

export type CreateComponentGenerationRunRequest = {
  files: File[];
  includeMarkdown?: boolean;
  maxCharsPerFile?: number;
  autoApprove?: boolean;
  runIntegration?: boolean;
  noLlm?: boolean;
  noStructuredOutput?: boolean;
  provider?: "ollama" | "openrouter";
  model?: string;
  temperature?: number;
  ollamaUrl?: string;
  openrouterUrl?: string;
  openrouterKey?: string;
  openrouterProvider?: string;
};

export type EventSubscriptionOptions = {
  sinceSeq?: number;
  onOpen?: () => void;
  onError?: (error: Event) => void;
};

async function parseError(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        return payload.detail;
      }
      return JSON.stringify(payload);
    } catch {
      return `HTTP ${response.status}`;
    }
  }
  try {
    const text = await response.text();
    return text || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

export async function createComponentGenerationRun(
  request: CreateComponentGenerationRunRequest
): Promise<ComponentGenerationRunSnapshot> {
  const formData = new FormData();
  request.files.forEach((file) => {
    formData.append("files", file, file.name);
  });
  formData.set("includeMarkdown", String(Boolean(request.includeMarkdown)));
  formData.set("maxCharsPerFile", String(request.maxCharsPerFile ?? 200000));
  formData.set("autoApprove", String(Boolean(request.autoApprove)));
  formData.set("runIntegration", String(Boolean(request.runIntegration)));
  formData.set("noLlm", String(Boolean(request.noLlm)));
  formData.set(
    "noStructuredOutput",
    String(Boolean(request.noStructuredOutput))
  );
  formData.set("provider", request.provider || "ollama");
  formData.set("model", request.model || "qwen3:14b");
  formData.set("temperature", String(request.temperature ?? 0));
  formData.set("ollamaUrl", request.ollamaUrl || "http://localhost:11434");
  formData.set(
    "openrouterUrl",
    request.openrouterUrl || "https://openrouter.ai/api/v1"
  );
  if (request.openrouterKey) {
    formData.set("openrouterKey", request.openrouterKey);
  }
  if (request.openrouterProvider) {
    formData.set("openrouterProvider", request.openrouterProvider);
  }

  const response = await fetch(`${API_BASE}/component-generation/runs`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as ComponentGenerationRunSnapshot;
}

export async function fetchComponentGenerationRun(
  runId: string
): Promise<ComponentGenerationRunSnapshot> {
  const response = await fetch(`${API_BASE}/component-generation/runs/${runId}`);
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as ComponentGenerationRunSnapshot;
}

export async function submitComponentGenerationDecision(
  runId: string,
  payload: { approved: boolean; feedback?: string }
): Promise<ComponentGenerationRunSnapshot> {
  const response = await fetch(
    `${API_BASE}/component-generation/runs/${runId}/decision`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as ComponentGenerationRunSnapshot;
}

export async function cancelComponentGenerationRun(
  runId: string
): Promise<ComponentGenerationRunSnapshot> {
  const response = await fetch(`${API_BASE}/component-generation/runs/${runId}/cancel`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as ComponentGenerationRunSnapshot;
}

const EVENT_TYPES = [
  "run_queued",
  "run_started",
  "node_started",
  "node_completed",
  "node_failed",
  "plan_proposed",
  "waiting_decision",
  "decision_submitted",
  "decision_received",
  "resumed",
  "run_finished",
  "run_failed",
  "run_canceled",
];

export function subscribeComponentGenerationEvents(
  runId: string,
  onEvent: (event: ComponentGenerationRunEvent) => void,
  options?: EventSubscriptionOptions
): EventSource {
  const url = buildApiUrl(`/component-generation/runs/${runId}/events`);
  if (options?.sinceSeq && options.sinceSeq > 0) {
    url.searchParams.set("sinceSeq", String(options.sinceSeq));
  }

  const source = new EventSource(url.toString());
  const handleMessage = (message: MessageEvent<string>) => {
    try {
      const parsed = JSON.parse(message.data) as ComponentGenerationRunEvent;
      if (typeof parsed.seq !== "number" || typeof parsed.type !== "string") {
        return;
      }
      onEvent(parsed);
    } catch {
      // Ignore malformed chunks from the stream.
    }
  };

  source.onopen = () => {
    options?.onOpen?.();
  };
  source.onerror = (error) => {
    options?.onError?.(error);
  };

  source.onmessage = handleMessage;
  EVENT_TYPES.forEach((type) => {
    source.addEventListener(type, handleMessage as EventListener);
  });

  return source;
}
