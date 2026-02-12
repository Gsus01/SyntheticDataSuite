"use client";

import Link from "next/link";
import React from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Edge,
  MarkerType,
  MiniMap,
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

import {
  COMPONENT_GENERATION_FORM_STORAGE_KEY,
  DEFAULT_FORM,
  GRAPH_EDGES_BASE,
  GRAPH_LAYOUT,
  MAX_EVENT_ENTRIES,
  MAX_LOG_ENTRIES,
} from "./component-generation/constants";
import { HitlReviewPanel } from "./component-generation/HitlReviewPanel";
import { IntegrationConfirmPanel } from "./component-generation/IntegrationConfirmPanel";
import { ToggleSwitch } from "./component-generation/ToggleSwitch";
import {
  Badge,
  Button,
  Input,
  Label,
  Panel,
  SectionTitle,
  Select,
} from "./component-generation/ui/primitives";
import { ArtifactsEventsPanel } from "./component-generation/ArtifactsEventsPanel";
import { graphNodeTypes } from "./component-generation/WorkflowGraphNode";
import type {
  FormState,
  LogEntry,
  LogFilter,
  UiNotice,
  WorkflowGraphNodeType,
  WorkflowNodeState,
} from "./component-generation/types";
import {
  edgeColorFor,
  formatBytes,
  formatTimestamp,
  isTerminal,
  levelClass,
  parsePersistedFormState,
  parseIntegrationSummaryForDisplay,
  parsePlanForDisplay,
  parseSnapshotLogLine,
  runStatusBadge,
  shouldIncludeLog,
  updateRunFromEvent,
  workflowNodeStateFor,
} from "./component-generation/utils";

type IntegrationUploadDialogState = {
  runId: string;
  registeredComponents: Array<{ name: string; version: string }>;
  builtImages: string[];
};

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
  const [mounted, setMounted] = React.useState(false);
  const [notice, setNotice] = React.useState<UiNotice | null>(null);
  const [confirmCancelAll, setConfirmCancelAll] = React.useState(false);
  const [integrationUploadDialog, setIntegrationUploadDialog] =
    React.useState<IntegrationUploadDialogState | null>(null);

  const eventSourceRef = React.useRef<EventSource | null>(null);
  const logContainerRef = React.useRef<HTMLDivElement | null>(null);
  const nextLogIdRef = React.useRef(1);
  const noticeTimeoutRef = React.useRef<number | null>(null);
  const shownIntegrationDialogRunsRef = React.useRef<Set<string>>(new Set());

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

      const loadedState = parsePersistedFormState(JSON.parse(rawStoredState));
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
      const payload = {
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
            ? run?.decisionStage === "integration"
              ? "Waiting for integration confirmation"
              : "Waiting for plan approval"
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
          fill: isDark ? "#cbd5e1" : "#334155",
          fontSize: 11,
          fontWeight: 600,
        },
      };
    });
  }, [graphNodes, isDark]);

  const filteredLogs = React.useMemo(
    () => logEntries.filter((entry) => shouldIncludeLog(entry, logFilter)),
    [logEntries, logFilter]
  );

  const planViewModel = React.useMemo(
    () => parsePlanForDisplay(run?.pendingPlan),
    [run?.pendingPlan]
  );
  const integrationSummary = React.useMemo(
    () => parseIntegrationSummaryForDisplay(run?.pendingIntegration),
    [run?.pendingIntegration]
  );
  const decisionStage = React.useMemo<"plan" | "integration" | null>(() => {
    if (!run?.awaitingDecision) return null;
    if (run.decisionStage === "integration") return "integration";
    if (run.decisionStage === "plan") return "plan";
    if (run.pendingIntegration) return "integration";
    return "plan";
  }, [run?.awaitingDecision, run?.decisionStage, run?.pendingIntegration]);

  React.useEffect(() => {
    if (!run?.runId) return;
    if (run.status !== "succeeded" || run.integrationStatus !== "completed") return;
    if (shownIntegrationDialogRunsRef.current.has(run.runId)) return;

    const result =
      run.integrationResult && typeof run.integrationResult === "object"
        ? (run.integrationResult as Record<string, unknown>)
        : null;

    const registeredRaw = Array.isArray(result?.registeredComponents)
      ? result.registeredComponents
      : [];
    const registeredComponents = registeredRaw
      .map((item) => {
        if (!item || typeof item !== "object" || Array.isArray(item)) return null;
        const row = item as Record<string, unknown>;
        const name =
          typeof row.name === "string" && row.name.trim()
            ? row.name.trim()
            : "(sin nombre)";
        const version =
          typeof row.version === "string" && row.version.trim()
            ? row.version.trim()
            : "(sin versión)";
        return { name, version };
      })
      .filter((item): item is { name: string; version: string } => Boolean(item));

    const builtImagesRaw = Array.isArray(result?.builtImages)
      ? result.builtImages
      : [];
    const builtImages = builtImagesRaw.filter(
      (item): item is string => typeof item === "string" && item.trim().length > 0
    );

    shownIntegrationDialogRunsRef.current.add(run.runId);
    setIntegrationUploadDialog({
      runId: run.runId,
      registeredComponents,
      builtImages,
    });
  }, [run]);

  const startRun = React.useCallback(async () => {
    if (!selectedFiles.length) {
      setError("Selecciona al menos un archivo de entrada.");
      return;
    }
    setStarting(true);
    setError(null);
    setIntegrationUploadDialog(null);
    try {
      const snapshot = await createComponentGenerationRun({
        files: selectedFiles,
        includeMarkdown: form.includeMarkdown,
        maxCharsPerFile: form.maxCharsPerFile,
        runIntegration: true,
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
    async (approved: boolean, stage?: "plan" | "integration") => {
      if (!run) return;
      setDecisionLoading(true);
      setError(null);
      try {
        const resolvedStage =
          stage ||
          (run.decisionStage === "integration" ? "integration" : "plan");
        const snapshot = await submitComponentGenerationDecision(run.runId, {
          approved,
          feedback:
            approved || resolvedStage === "integration" ? "" : feedback,
          stage: resolvedStage,
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

  const isAwaitingDecision = Boolean(run?.awaitingDecision);
  const runBadgeVariant = run ? runStatusBadge(run.status) : "neutral";

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-slate-200 bg-white/90 px-5 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="inline-flex rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            Volver al editor
          </Link>
          <div>
            <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Generación Automática de Componentes
            </h1>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Logs en vivo, grafo y revisión HITL.
            </p>
          </div>
        </div>
        <div className="relative flex items-center gap-2">
          {run ? (
            <Badge variant={runBadgeVariant}>
              {run.status.replace("_", " ")}
            </Badge>
          ) : (
            <span className="text-xs text-slate-500 dark:text-slate-400">Sin ejecución activa</span>
          )}
          <Badge variant={streamConnected ? "success" : "neutral"}>
            {streamConnected ? "SSE online" : "SSE offline"}
          </Badge>
          <Button
            type="button"
            onClick={() => {
              setNotice(null);
              setConfirmCancelAll(true);
            }}
            disabled={cancelAllLoading}
            variant="warning"
            size="sm"
            title="Interrumpe todas las ejecuciones activas en background"
          >
            {cancelAllLoading ? "Cerrando..." : "Cerrar sesiones"}
          </Button>
          <Button
            type="button"
            onClick={toggleTheme}
            variant="secondary"
            size="sm"
            aria-label="Cambiar tema claro/oscuro"
            suppressHydrationWarning
          >
            {!mounted ? <span className="opacity-0">☀️</span> : isDark ? "☀️" : "🌙"}
          </Button>

          {confirmCancelAll ? (
            <div className="absolute right-0 top-full z-50 mt-2 w-[min(92vw,26rem)]">
              <div className="rounded-xl border border-amber-300 bg-amber-50 px-3 py-2 shadow-xl dark:border-amber-700 dark:bg-amber-900/30">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-[12px] leading-snug text-slate-700 dark:text-slate-200">
                    Esto interrumpirá todas las sesiones activas en background.
                  </p>
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      onClick={() => setConfirmCancelAll(false)}
                      variant="secondary"
                      size="sm"
                    >
                      Volver
                    </Button>
                    <Button
                      type="button"
                      onClick={() => void cancelAllRuns()}
                      disabled={cancelAllLoading}
                      variant="warning"
                      size="sm"
                    >
                      {cancelAllLoading ? "Cerrando..." : "Sí, cerrar"}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {notice && !confirmCancelAll ? (
            <div className="absolute right-0 top-full z-50 mt-2 w-[min(90vw,24rem)]">
              <div
                className={`flex items-center justify-between gap-3 rounded-xl border px-3 py-2 text-[13px] leading-snug shadow-xl ${notice.kind === "success"
                    ? "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200"
                    : "border-slate-300 bg-white text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                  }`}
                role="status"
                aria-live="polite"
              >
                <span>{notice.message}</span>
                <Button type="button" variant="secondary" size="sm" onClick={() => setNotice(null)}>
                  Cerrar
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      </header>

      <div className="grid h-[calc(100vh-4rem)] grid-cols-1 lg:grid-cols-[360px_1fr]">
        <aside className="flex min-h-0 flex-col overflow-y-auto border-r border-slate-200 bg-slate-50/80 p-4 dark:border-slate-800 dark:bg-slate-900/30">
          <div className="space-y-4">
            <Panel>
              <SectionTitle>Archivos de entrada</SectionTitle>
              <label className="mt-3 flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-slate-300 bg-white px-3 py-4 text-center text-xs text-slate-700 transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800">
                <span className="font-medium">Seleccionar archivos</span>
                <span className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                  .py, .ipynb, .md, etc.
                </span>
                <input
                  type="file"
                  multiple
                  accept=".py,.ipynb,.md,.txt,.json,.yaml,.yml,.toml,.ini,.cfg,.csv,.ts,.tsx,.js,.jsx"
                  className="hidden"
                  onChange={(event) => {
                    const files = Array.from(event.target.files || []);
                    setSelectedFiles(files);
                  }}
                />
              </label>
              <div className="mt-3 max-h-40 space-y-2 overflow-y-auto">
                {selectedFiles.length === 0 ? (
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">Sin archivos seleccionados.</p>
                ) : (
                  selectedFiles.map((file) => (
                    <div
                      key={`${file.name}-${file.size}`}
                      className="rounded border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] dark:border-slate-700 dark:bg-slate-900/50"
                    >
                      <div className="truncate font-medium text-slate-700 dark:text-slate-200">{file.name}</div>
                      <div className="text-slate-500 dark:text-slate-400">{formatBytes(file.size)}</div>
                    </div>
                  ))
                )}
              </div>
            </Panel>

            <Panel>
              <SectionTitle>Configuración</SectionTitle>
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
                  <Label>Max chars por fichero</Label>
                  <Input
                    type="number"
                    min={1}
                    value={form.maxCharsPerFile}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        maxCharsPerFile: Number(event.target.value) || 1,
                      }))
                    }
                  />
                </label>

                <label className="block">
                  <Label>Provider</Label>
                  <Select
                    value={form.provider}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        provider: event.target.value as "ollama" | "openrouter",
                      }))
                    }
                  >
                    <option value="ollama">ollama</option>
                    <option value="openrouter">openrouter</option>
                  </Select>
                </label>

                <label className="block">
                  <Label>Modelo</Label>
                  <Input
                    type="text"
                    value={form.model}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, model: event.target.value }))
                    }
                  />
                </label>

                <label className="block">
                  <Label>Temperatura</Label>
                  <Input
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
                    <Label>Ollama URL</Label>
                    <Input
                      type="text"
                      value={form.ollamaUrl}
                      onChange={(event) =>
                        setForm((prev) => ({ ...prev, ollamaUrl: event.target.value }))
                      }
                    />
                  </label>
                ) : (
                  <>
                    <label className="block">
                      <Label>OpenRouter URL</Label>
                      <Input
                        type="text"
                        value={form.openrouterUrl}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterUrl: event.target.value,
                          }))
                        }
                      />
                    </label>
                    <label className="block">
                      <Label>OpenRouter Key</Label>
                      <Input
                        type="password"
                        value={form.openrouterKey}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterKey: event.target.value,
                          }))
                        }
                      />
                    </label>
                    <ToggleSwitch
                      label="Recordar API key en este navegador"
                      checked={rememberOpenrouterKey}
                      onChange={(checked) => setRememberOpenrouterKey(checked)}
                      className="rounded-md border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-700 dark:bg-slate-900"
                      labelClassName="text-[11px]"
                    />
                    <label className="block">
                      <Label>Provider order</Label>
                      <Input
                        type="text"
                        value={form.openrouterProvider}
                        onChange={(event) =>
                          setForm((prev) => ({
                            ...prev,
                            openrouterProvider: event.target.value,
                          }))
                        }
                        placeholder="openai,anthropic"
                      />
                    </label>
                  </>
                )}

                <Button
                  type="button"
                  onClick={resetLocalSettings}
                  variant="secondary"
                  className="w-full"
                >
                  Restablecer ajustes locales
                </Button>
              </div>
            </Panel>

            <Panel>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  onClick={startRun}
                  disabled={starting || (run ? !isTerminal(run.status) : false)}
                  variant="primary"
                  className="flex-1"
                >
                  {starting ? "Iniciando..." : "Iniciar generación"}
                </Button>
                <Button
                  type="button"
                  onClick={cancelRun}
                  disabled={!run || !run.canCancel || cancelLoading}
                  variant="danger"
                  className="flex-1"
                >
                  {cancelLoading ? "Deteniendo..." : "Detener"}
                </Button>
              </div>
              {error ? (
                <p className="mt-2 text-[11px] text-red-700 dark:text-red-300">{error}</p>
              ) : null}
            </Panel>
          </div>
        </aside>

        <main className="flex min-h-0 flex-col">
          <section className="h-[44%] border-b border-slate-200 px-4 py-3 dark:border-slate-800">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
              LangGraph en tiempo real
            </h2>
            <div className="h-[calc(100%-1.8rem)] overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
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
                <Background color={isDark ? "#334155" : "#cbd5e1"} gap={28} size={1} />
                <MiniMap
                  pannable
                  zoomable
                  className="!border !border-slate-300 !bg-white dark:!border-slate-700 dark:!bg-slate-900"
                  maskColor={isDark ? "rgba(2, 6, 23, 0.55)" : "rgba(148, 163, 184, 0.3)"}
                />
                <Controls className="!border !border-slate-300 dark:!border-slate-700" />
              </ReactFlow>
            </div>
          </section>

          <section className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden p-4 xl:grid-cols-[0.95fr_1.45fr]">
            <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
              <Panel>
                <SectionTitle>Estado de ejecución</SectionTitle>
                {run ? (
                  <div className="mt-2 space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Run ID</span>
                      <span className="font-mono text-[11px] text-slate-700 dark:text-slate-200">{run.runId}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Estado</span>
                      <Badge variant={runBadgeVariant}>
                        {run.status.replace("_", " ")}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Actualizado</span>
                      <span className="text-slate-700 dark:text-slate-200">{formatTimestamp(run.updatedAt)}</span>
                    </div>
                    <div className="pt-1 text-[11px] text-slate-500 dark:text-slate-400">
                      Session dir:
                      <div className="mt-1 break-all font-mono text-slate-700 dark:text-slate-200">
                        {run.sessionDir}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                    Aún no has lanzado ninguna ejecución.
                  </p>
                )}
              </Panel>

              <ArtifactsEventsPanel
                run={run}
                events={events}
                streamConnected={streamConnected}
              />
            </div>

            <Panel
              className={`min-h-0 flex flex-col p-0 ${isAwaitingDecision
                  ? "border-amber-300 bg-amber-50/60 dark:border-amber-700 dark:bg-amber-900/10"
                  : ""
                }`}
            >
              {run?.awaitingDecision ? (
                decisionStage === "integration" ? (
                  <IntegrationConfirmPanel
                    run={run}
                    summary={integrationSummary}
                    decisionLoading={decisionLoading}
                    onApprove={() => void submitDecision(true, "integration")}
                    onReject={() => void submitDecision(false, "integration")}
                  />
                ) : (
                  <HitlReviewPanel
                    planViewModel={planViewModel}
                    feedback={feedback}
                    onFeedbackChange={setFeedback}
                    onApprove={() => void submitDecision(true, "plan")}
                    onRequestChanges={() => void submitDecision(false, "plan")}
                    decisionLoading={decisionLoading}
                  />
                )
              ) : (
                <>
                  <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-3 py-2 dark:border-slate-700">
                    <SectionTitle className="shrink-0">Consola de ejecución</SectionTitle>
                    <div className="ml-auto flex min-w-0 items-center justify-end gap-2 overflow-x-auto text-[11px]">
                      <Select
                        fullWidth={false}
                        value={logFilter}
                        onChange={(event) => setLogFilter(event.target.value as LogFilter)}
                        className="h-8 min-w-28 shrink-0 px-2 py-0 text-[11px]"
                      >
                        <option value="ALL">all</option>
                        <option value="ERROR">error</option>
                        <option value="WARNING">warning</option>
                        <option value="INFO">info</option>
                        <option value="DEBUG">debug</option>
                        <option value="STDOUT">stdout</option>
                        <option value="STDERR">stderr</option>
                      </Select>
                      <Button
                        type="button"
                        onClick={() => setAutoScrollLogs((prev) => !prev)}
                        variant="secondary"
                        size="sm"
                        className="shrink-0 whitespace-nowrap"
                      >
                        {autoScrollLogs ? "autoscroll on" : "autoscroll off"}
                      </Button>
                      <Button
                        type="button"
                        onClick={() => setLogEntries([])}
                        variant="secondary"
                        size="sm"
                        className="shrink-0"
                      >
                        clear
                      </Button>
                    </div>
                  </div>
                  <div
                    ref={logContainerRef}
                    className="min-h-0 flex-1 overflow-y-auto bg-slate-50 p-2 font-mono text-[11px] whitespace-pre-wrap break-words [scrollbar-gutter:stable] dark:bg-slate-950/60"
                  >
                    {filteredLogs.length === 0 ? (
                      <p className="text-slate-500 dark:text-slate-400">Sin logs todavía.</p>
                    ) : (
                      filteredLogs.map((entry) => (
                        <div key={entry.id} className={`leading-5 ${levelClass(entry.level)}`}>
                          <span className="text-slate-500 dark:text-slate-400">[{entry.level}]</span>{" "}
                          <span className="text-slate-500 dark:text-slate-400">{entry.source}:</span>{" "}
                          {entry.line}
                        </div>
                      ))
                    )}
                  </div>
                </>
              )}
            </Panel>
          </section>
        </main>
      </div>

      {integrationUploadDialog ? (
        <div className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-900/45 p-4 backdrop-blur-[1px]">
          <div className="w-[min(94vw,34rem)] rounded-2xl border border-emerald-300 bg-white p-4 shadow-2xl dark:border-emerald-700 dark:bg-slate-900">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">
                  Componentes Subidos Correctamente
                </h3>
                <p className="mt-1 text-[12px] text-slate-600 dark:text-slate-300">
                  Run <span className="font-mono">{integrationUploadDialog.runId}</span>
                </p>
              </div>
              <Badge variant="success">OK</Badge>
            </div>

            <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3 text-[12px] dark:border-slate-700 dark:bg-slate-800/50">
              <div className="font-semibold text-slate-800 dark:text-slate-100">
                {integrationUploadDialog.registeredComponents.length} componente(s) registrado(s)
              </div>
            </div>

            <div className="mt-3 space-y-2">
              {integrationUploadDialog.registeredComponents.length === 0 ? (
                <p className="text-[12px] text-slate-500 dark:text-slate-400">
                  No se recibió detalle de componentes en la respuesta de integración.
                </p>
              ) : (
                <div className="max-h-48 space-y-1 overflow-y-auto [scrollbar-gutter:stable]">
                  {integrationUploadDialog.registeredComponents.map((component) => (
                    <div
                      key={`${component.name}-${component.version}`}
                      className="flex items-center justify-between rounded border border-slate-200 bg-slate-50 px-2 py-1.5 text-[12px] dark:border-slate-700 dark:bg-slate-800/40"
                    >
                      <span className="font-mono text-slate-700 dark:text-slate-200">
                        {component.name}
                      </span>
                      <span className="text-slate-500 dark:text-slate-400">
                        {component.version}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {integrationUploadDialog.builtImages.length > 0 ? (
              <details className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-2 text-[11px] dark:border-slate-700 dark:bg-slate-800/40">
                <summary className="cursor-pointer font-semibold text-slate-600 dark:text-slate-300">
                  Imágenes construidas ({integrationUploadDialog.builtImages.length})
                </summary>
                <div className="mt-2 max-h-24 space-y-1 overflow-y-auto font-mono text-slate-600 [scrollbar-gutter:stable] dark:text-slate-300">
                  {integrationUploadDialog.builtImages.map((image) => (
                    <div key={image}>{image}</div>
                  ))}
                </div>
              </details>
            ) : null}

            <div className="mt-4 flex justify-end">
              <Button
                type="button"
                variant="primary"
                size="sm"
                onClick={() => setIntegrationUploadDialog(null)}
              >
                Cerrar
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
