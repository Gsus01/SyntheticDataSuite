"use client";

import Link from "next/link";
import React from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  MarkerType,
  Node,
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

const GRAPH_NODES = [
  { id: "load", label: "load", x: 40, y: 120 },
  { id: "analyst", label: "analyst", x: 240, y: 120 },
  { id: "hitl", label: "hitl", x: 440, y: 120 },
  { id: "developer", label: "developer", x: 640, y: 120 },
  { id: "tester", label: "tester", x: 840, y: 120 },
  { id: "integration", label: "integration", x: 1040, y: 120 },
  { id: "repair", label: "repair", x: 840, y: 280 },
];

const GRAPH_EDGES: Edge[] = [
  {
    id: "load-analyst",
    source: "load",
    target: "analyst",
    markerEnd: { type: MarkerType.ArrowClosed },
  },
  {
    id: "analyst-hitl",
    source: "analyst",
    target: "hitl",
    markerEnd: { type: MarkerType.ArrowClosed },
  },
  {
    id: "hitl-analyst",
    source: "hitl",
    target: "analyst",
    markerEnd: { type: MarkerType.ArrowClosed },
    label: "revise",
    style: { strokeDasharray: "5 4" },
  },
  {
    id: "hitl-developer",
    source: "hitl",
    target: "developer",
    markerEnd: { type: MarkerType.ArrowClosed },
  },
  {
    id: "developer-tester",
    source: "developer",
    target: "tester",
    markerEnd: { type: MarkerType.ArrowClosed },
  },
  {
    id: "tester-repair",
    source: "tester",
    target: "repair",
    markerEnd: { type: MarkerType.ArrowClosed },
    label: "needs_fix",
    style: { strokeDasharray: "5 4" },
  },
  {
    id: "repair-tester",
    source: "repair",
    target: "tester",
    markerEnd: { type: MarkerType.ArrowClosed },
  },
  {
    id: "tester-integration",
    source: "tester",
    target: "integration",
    markerEnd: { type: MarkerType.ArrowClosed },
  },
];

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

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled"]);

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

function statusColor(state: string) {
  if (state === "running") {
    return {
      border: "border-sky-500",
      bg: "bg-sky-500/15",
      text: "text-sky-200",
    };
  }
  if (state === "completed") {
    return {
      border: "border-emerald-500",
      bg: "bg-emerald-500/15",
      text: "text-emerald-200",
    };
  }
  if (state === "failed") {
    return {
      border: "border-rose-500",
      bg: "bg-rose-500/20",
      text: "text-rose-200",
    };
  }
  if (state === "canceled") {
    return {
      border: "border-amber-500",
      bg: "bg-amber-500/15",
      text: "text-amber-200",
    };
  }
  return {
    border: "border-slate-600",
    bg: "bg-slate-800/50",
    text: "text-slate-200",
  };
}

function runStatusBadge(status: string): string {
  if (status === "queued") return "bg-slate-700 text-slate-100";
  if (status === "running") return "bg-sky-600 text-white";
  if (status === "waiting_decision") return "bg-amber-500 text-slate-950";
  if (status === "succeeded") return "bg-emerald-600 text-white";
  if (status === "failed") return "bg-rose-600 text-white";
  if (status === "canceled") return "bg-amber-600 text-white";
  return "bg-slate-700 text-slate-100";
}

function updateRunFromEvent(
  previous: ComponentGenerationRunSnapshot,
  event: ComponentGenerationRunEvent
): ComponentGenerationRunSnapshot {
  const next: ComponentGenerationRunSnapshot = {
    ...previous,
    nodeStates: { ...previous.nodeStates },
    updatedAt: event.timestamp || previous.updatedAt,
    lastSeq: Math.max(previous.lastSeq ?? 0, event.seq ?? 0),
  };
  const payload = event.payload || {};

  if (event.type === "run_started") {
    next.status = "running";
  } else if (event.type === "waiting_decision") {
    next.status = "waiting_decision";
    const plan = payload.plan;
    if (plan && typeof plan === "object") {
      next.pendingPlan = plan as Record<string, unknown>;
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
  } else if (event.type === "plan_proposed") {
    const plan = payload.plan;
    if (plan && typeof plan === "object") {
      next.pendingPlan = plan as Record<string, unknown>;
    }
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

function eventSummary(event: ComponentGenerationRunEvent): string {
  const payload = event.payload || {};
  if (event.type === "run_queued") return "Run en cola";
  if (event.type === "run_started") return "Ejecución iniciada";
  if (event.type === "waiting_decision") return "Esperando aprobación del plan";
  if (event.type === "resumed") return "Reanudado tras decisión HITL";
  if (event.type === "decision_submitted") {
    return payload.approved ? "Decisión enviada: aprobar" : "Decisión enviada: revisar";
  }
  if (event.type === "run_finished") return "Run completado";
  if (event.type === "run_failed") {
    return `Run fallido: ${typeof payload.error === "string" ? payload.error : "error"}`;
  }
  if (event.type === "run_canceled") return "Run cancelado por usuario";
  if (event.type === "node_started") {
    return `Nodo iniciado: ${typeof payload.node === "string" ? payload.node : "n/a"}`;
  }
  if (event.type === "node_completed") {
    return `Nodo completado: ${typeof payload.node === "string" ? payload.node : "n/a"}`;
  }
  if (event.type === "node_failed") {
    const node = typeof payload.node === "string" ? payload.node : "n/a";
    const error = typeof payload.error === "string" ? payload.error : "error";
    return `Nodo fallido: ${node} (${error})`;
  }
  if (event.type === "plan_proposed") return "Plan propuesto recibido";
  return `${event.type}: ${JSON.stringify(payload)}`;
}

export default function ComponentGenerationPage() {
  const [selectedFiles, setSelectedFiles] = React.useState<File[]>([]);
  const [form, setForm] = React.useState<FormState>(DEFAULT_FORM);
  const [run, setRun] = React.useState<ComponentGenerationRunSnapshot | null>(null);
  const [events, setEvents] = React.useState<ComponentGenerationRunEvent[]>([]);
  const [feedback, setFeedback] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [starting, setStarting] = React.useState(false);
  const [decisionLoading, setDecisionLoading] = React.useState(false);
  const [cancelLoading, setCancelLoading] = React.useState(false);
  const [streamConnected, setStreamConnected] = React.useState(false);

  const eventSourceRef = React.useRef<EventSource | null>(null);

  const closeStream = React.useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStreamConnected(false);
  }, []);

  const refreshRun = React.useCallback(async (runId: string) => {
    const snapshot = await fetchComponentGenerationRun(runId);
    setRun(snapshot);
    if (isTerminal(snapshot.status)) {
      closeStream();
    }
  }, [closeStream]);

  const connectStream = React.useCallback(
    (runId: string, sinceSeq: number) => {
      closeStream();
      const source = subscribeComponentGenerationEvents(
        runId,
        (event) => {
          setEvents((prev) => {
            const next = [...prev, event];
            if (next.length > 600) {
              return next.slice(next.length - 600);
            }
            return next;
          });
          setRun((prev) => (prev ? updateRunFromEvent(prev, event) : prev));
          if (event.type === "run_finished" || event.type === "run_failed" || event.type === "run_canceled") {
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
    [closeStream]
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
        // Keep stream events as source of truth if polling fails.
      });
    }, 2000);
    return () => clearInterval(timer);
  }, [run?.runId, run?.status, refreshRun]);

  const graphNodes = React.useMemo<Node[]>(
    () =>
      GRAPH_NODES.map((node) => {
        const runtime = run?.nodeStates?.[node.id];
        const state = runtime?.state || "pending";
        const colors = statusColor(state);
        const message = runtime?.message;
        return {
          id: node.id,
          position: { x: node.x, y: node.y },
          data: {
            label: (
              <div className="space-y-1">
                <div className="font-semibold text-xs uppercase tracking-wide">{node.label}</div>
                <div className="text-[10px] uppercase">{state}</div>
                {message ? (
                  <div className="text-[10px] leading-snug opacity-90">{message}</div>
                ) : null}
              </div>
            ),
          },
          style: {
            minWidth: 150,
            borderRadius: 10,
            borderWidth: 1,
            borderStyle: "solid",
          },
          className: `${colors.border} ${colors.bg} ${colors.text}`,
          draggable: false,
          selectable: false,
        };
      }),
    [run]
  );

  const plan = run?.pendingPlan;
  const planComponents = React.useMemo(() => {
    if (!plan || typeof plan !== "object") return [];
    const components = (plan as { components?: unknown }).components;
    if (!Array.isArray(components)) return [];
    return components
      .map((component) => {
        if (!component || typeof component !== "object") return null;
        const item = component as Record<string, unknown>;
        return {
          name: typeof item.name === "string" ? item.name : "component",
          type: typeof item.type === "string" ? item.type : "other",
          description:
            typeof item.description === "string" ? item.description : "",
        };
      })
      .filter((item): item is { name: string; type: string; description: string } => Boolean(item));
  }, [plan]);

  const handleStart = React.useCallback(async () => {
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
      connectStream(snapshot.runId, snapshot.lastSeq);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "No se pudo iniciar la generación.";
      setError(message);
    } finally {
      setStarting(false);
    }
  }, [connectStream, form, selectedFiles]);

  const handleDecision = React.useCallback(
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
        const message =
          err instanceof Error ? err.message : "No se pudo enviar la decisión.";
        setError(message);
      } finally {
        setDecisionLoading(false);
      }
    },
    [feedback, run]
  );

  const handleCancel = React.useCallback(async () => {
    if (!run) return;
    setCancelLoading(true);
    setError(null);
    try {
      const snapshot = await cancelComponentGenerationRun(run.runId);
      setRun(snapshot);
      closeStream();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "No se pudo interrumpir la ejecución.";
      setError(message);
    } finally {
      setCancelLoading(false);
    }
  }, [closeStream, run]);

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
              LangGraph completo por API, con revisión HITL y cancelación.
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
              </div>
            </section>

            <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleStart}
                  disabled={starting || (run ? !isTerminal(run.status) : false)}
                  className="inline-flex flex-1 items-center justify-center rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-900/60"
                >
                  {starting ? "Iniciando..." : "Iniciar generación"}
                </button>
                <button
                  type="button"
                  onClick={handleCancel}
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
          <section className="h-[42%] border-b border-slate-800 px-4 py-3">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
              LangGraph en tiempo real
            </h2>
            <div className="h-[calc(100%-1.75rem)] rounded-lg border border-slate-800 bg-slate-900">
              <ReactFlow
                nodes={graphNodes}
                edges={GRAPH_EDGES}
                fitView
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                panOnDrag
                zoomOnScroll
                className="bg-slate-900"
              >
                <Background color="#334155" gap={22} size={1} />
                <Controls />
              </ReactFlow>
            </div>
          </section>

          <section className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 xl:grid-cols-[1.2fr_1fr]">
            <div className="flex min-h-0 flex-col gap-4">
              {run?.awaitingDecision ? (
                <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-amber-200">
                    Revisión HITL
                  </h3>
                  <p className="mt-1 text-xs text-amber-100">
                    El agente está esperando tu decisión sobre el plan propuesto.
                  </p>
                  <div className="mt-3 max-h-44 space-y-2 overflow-y-auto pr-1">
                    {planComponents.length === 0 ? (
                      <p className="text-[11px] text-amber-100/80">
                        No hay componentes listados en el plan.
                      </p>
                    ) : (
                      planComponents.map((component) => (
                        <div
                          key={`${component.type}-${component.name}`}
                          className="rounded border border-amber-300/20 bg-black/20 px-2 py-1.5 text-[11px]"
                        >
                          <div className="font-semibold text-amber-100">
                            {component.name}
                            <span className="ml-2 rounded bg-amber-300/20 px-1.5 py-0.5 text-[10px] uppercase">
                              {component.type}
                            </span>
                          </div>
                          {component.description ? (
                            <div className="mt-1 text-amber-100/80">
                              {component.description}
                            </div>
                          ) : null}
                        </div>
                      ))
                    )}
                  </div>
                  <label className="mt-3 block">
                    <span className="mb-1 block text-[11px] text-amber-100/80">
                      Feedback para revisar plan
                    </span>
                    <textarea
                      value={feedback}
                      onChange={(event) => setFeedback(event.target.value)}
                      rows={4}
                      className="w-full rounded border border-amber-300/20 bg-black/25 px-2 py-1.5 text-xs text-amber-50"
                      placeholder="Qué debería cambiar el agente..."
                    />
                  </label>
                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      onClick={() => void handleDecision(true)}
                      disabled={decisionLoading}
                      className="inline-flex items-center justify-center rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-emerald-900/50"
                    >
                      Aprobar plan
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleDecision(false)}
                      disabled={decisionLoading}
                      className="inline-flex items-center justify-center rounded-md bg-amber-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-amber-900/60 disabled:text-amber-200"
                    >
                      Solicitar cambios
                    </button>
                  </div>
                </div>
              ) : null}

              <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-slate-800 bg-slate-900">
                <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Timeline de eventos
                  </h3>
                  <span className="text-[11px] text-slate-500">
                    {events.length} eventos
                  </span>
                </div>
                <div className="min-h-0 flex-1 space-y-1 overflow-y-auto p-2 font-mono text-[11px]">
                  {events.length === 0 ? (
                    <p className="text-slate-500">Sin eventos todavía.</p>
                  ) : (
                    events.map((event) => (
                      <div
                        key={event.seq}
                        className="rounded border border-slate-800 bg-slate-950/60 px-2 py-1.5"
                      >
                        <div className="flex items-center justify-between text-[10px] text-slate-500">
                          <span>#{event.seq}</span>
                          <span>{formatTimestamp(event.timestamp)}</span>
                        </div>
                        <div className="mt-0.5 text-slate-200">
                          {eventSummary(event)}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            <div className="flex min-h-0 flex-col gap-4">
              <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
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

              <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
                  Artefactos y reporte
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
                    <div className="space-y-1">
                      <span className="text-slate-400">Review report</span>
                      <pre className="max-h-28 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-slate-300">
                        {run.reviewReport || "(sin reporte)"}
                      </pre>
                    </div>
                    <div className="space-y-1">
                      <span className="text-slate-400">Error</span>
                      <pre className="max-h-20 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] text-rose-300">
                        {run.error || "(sin error)"}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-slate-500">Sin artefactos todavía.</p>
                )}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
