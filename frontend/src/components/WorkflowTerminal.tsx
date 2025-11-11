"use client";

import React from "react";
import type { Node } from "reactflow";

import {
  fetchWorkflowLogs,
  type SubmitWorkflowResult,
  type WorkflowLogChunk,
} from "@/lib/workflow";
import type { FlowNodeData } from "@/types/flow";

type WorkflowTerminalProps = {
  isOpen: boolean;
  onToggle: () => void;
  workflowName?: string | null;
  namespace?: string | null;
  nodeSlugMap?: Record<string, string> | null;
  submitting: boolean;
  submitResult: SubmitWorkflowResult | null;
  submitError: string | null;
  nodes: Node<FlowNodeData>[];
};

type TerminalLine = {
  id: string;
  slug: string;
  label: string;
  color: string;
  podName: string;
  text: string;
  timestamp: number;
};

const MIN_HEIGHT = 180;
const MAX_HEIGHT = 560;
const COLLAPSED_HEIGHT = 44;
const MAX_LINES = 2000;

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function sanitizeSegment(value: string | undefined | null, fallback: string): string {
  const cleanValue = (value ?? "").trim();
  const candidate = cleanValue
    .replace(/[^0-9A-Za-z._-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "");
  const sanitizedCandidate = candidate || fallback.trim();
  const finalValue = sanitizedCandidate
    .replace(/[^0-9A-Za-z._-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "");
  return finalValue || "node";
}

function slugToColor(slug: string): string {
  let hash = 0;
  for (let i = 0; i < slug.length; i += 1) {
    hash = (hash * 31 + slug.charCodeAt(i)) & 0xffffffff;
  }
  const hue = Math.abs(hash) % 360;
  const saturation = 65;
  const lightness = 50;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function formatTimestamp(timestampMs: number): string {
  return timeFormatter.format(new Date(timestampMs));
}

const SCROLL_EPSILON = 32;

export default function WorkflowTerminal({
  isOpen,
  onToggle,
  workflowName,
  namespace,
  nodeSlugMap,
  submitting,
  submitResult,
  submitError,
  nodes,
}: WorkflowTerminalProps) {
  const resolvedNamespace = namespace ?? submitResult?.namespace ?? undefined;
  const [height, setHeight] = React.useState<number>(320);
  const [isResizing, setIsResizing] = React.useState(false);
  const [lines, setLines] = React.useState<TerminalLine[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [fetchError, setFetchError] = React.useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = React.useState<number | null>(null);
  const [hasUnread, setHasUnread] = React.useState(false);
  const [autoScroll, setAutoScroll] = React.useState(true);

  const processedChunksRef = React.useRef<Set<string>>(new Set());
  const cursorRef = React.useRef<string | null>(null);
  const scheduleRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelPollingRef = React.useRef(false);
  const scrollRef = React.useRef<HTMLDivElement | null>(null);
  const startYRef = React.useRef(0);
  const startHeightRef = React.useRef(0);

  const slugInfoMap = React.useMemo(() => {
    const map = new Map<
      string,
      {
        label: string;
        color: string;
      }
    >();

    const nodeById = new Map(nodes.map((node) => [node.id, node]));

    if (nodeSlugMap) {
      Object.entries(nodeSlugMap).forEach(([nodeId, slug]) => {
        const node = nodeById.get(nodeId);
        const label = node?.data.label || node?.data.templateName || slug;
        map.set(slug, {
          label: label ?? slug,
          color: slugToColor(slug),
        });
      });
    }

    nodes.forEach((node) => {
      const fallbackSlug = sanitizeSegment(node.data.templateName || node.data.label || node.id, node.id);
      if (!map.has(fallbackSlug)) {
        map.set(fallbackSlug, {
          label: node.data.label || node.data.templateName || fallbackSlug,
          color: slugToColor(fallbackSlug),
        });
      }
    });

    return map;
  }, [nodeSlugMap, nodes]);

  const appendChunks = React.useCallback(
    (chunks: WorkflowLogChunk[]) => {
      if (!chunks.length) return;

      let mutated = false;
      setLines((prev) => {
        const next = [...prev];
        for (const chunk of chunks) {
          const fingerprint = `${chunk.key}:${chunk.endOffset}`;
          if (processedChunksRef.current.has(fingerprint)) {
            continue;
          }
          processedChunksRef.current.add(fingerprint);

          const info = slugInfoMap.get(chunk.nodeSlug);
          const label = info?.label ?? chunk.nodeSlug;
          const color = info?.color ?? "hsl(210, 18%, 66%)";
          const timestampMs = Math.round((chunk.timestamp ?? Date.now() / 1000) * 1000);
          const normalized = chunk.content.replace(/\r\n/g, "\n");
          const linesContent = normalized.split("\n");
          const baseId = `${chunk.key}:${chunk.startOffset}`;

          linesContent.forEach((text, index) => {
            const isLastLine = index === linesContent.length - 1;
            if (isLastLine && text === "") {
              return;
            }
            const id = `${baseId}:${index}`;
            next.push({
              id,
              slug: chunk.nodeSlug,
              label,
              color,
              podName: chunk.podName,
              text,
              timestamp: timestampMs,
            });
            mutated = true;
          });
        }

        if (!mutated) {
          return prev;
        }

        if (next.length > MAX_LINES) {
          next.splice(0, next.length - MAX_LINES);
        }

        return next;
      });

      if (mutated) {
        setLastUpdated(Date.now());
        if (!isOpen) {
          setHasUnread(true);
        }
      }
    },
    [isOpen, slugInfoMap]
  );

  const pollLogs = React.useCallback(async () => {
    if (!workflowName) {
      return;
    }

    if (scheduleRef.current) {
      clearTimeout(scheduleRef.current);
      scheduleRef.current = null;
    }

    setLoading(true);
    try {
      const result = await fetchWorkflowLogs({
        workflowName,
        namespace: resolvedNamespace,
        cursor: cursorRef.current ?? undefined,
      });

      cursorRef.current = result.cursor;
      if (result.chunks.length) {
        appendChunks(result.chunks);
      }
      setFetchError(null);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Error obteniendo logs de la ejecución";
      setFetchError(message);
    } finally {
      setLoading(false);

      if (!cancelPollingRef.current && workflowName) {
        scheduleRef.current = setTimeout(() => {
          pollLogs().catch(() => undefined);
        }, 4000);
      }
    }
  }, [appendChunks, resolvedNamespace, workflowName]);

  React.useEffect(() => {
    processedChunksRef.current.clear();
    cursorRef.current = null;
    setLines([]);
    setFetchError(null);
    setLastUpdated(null);
    setHasUnread(false);
  }, [workflowName]);

  React.useEffect(() => {
    if (!workflowName) {
      cancelPollingRef.current = true;
      if (scheduleRef.current) {
        clearTimeout(scheduleRef.current);
        scheduleRef.current = null;
      }
      return;
    }

    cancelPollingRef.current = false;
    pollLogs().catch(() => undefined);

    return () => {
      cancelPollingRef.current = true;
      if (scheduleRef.current) {
        clearTimeout(scheduleRef.current);
        scheduleRef.current = null;
      }
    };
  }, [pollLogs, workflowName]);

  React.useEffect(() => {
    if (isOpen) {
      setHasUnread(false);
    }
  }, [isOpen]);

  React.useEffect(() => {
    if (!autoScroll) return;
    const container = scrollRef.current;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
  }, [lines, autoScroll]);

  React.useEffect(() => {
    if (isOpen && height < MIN_HEIGHT) {
      setHeight(MIN_HEIGHT);
    }
  }, [height, isOpen]);

  const handleResizeStart = React.useCallback((event: React.MouseEvent) => {
    if (!isOpen) return;
    setIsResizing(true);
    startYRef.current = event.clientY;
    startHeightRef.current = height;
    event.preventDefault();
  }, [height, isOpen]);

  React.useEffect(() => {
    if (!isResizing) return;

    const handleMove = (event: MouseEvent) => {
      const delta = startYRef.current - event.clientY;
      const nextHeight = Math.min(
        MAX_HEIGHT,
        Math.max(MIN_HEIGHT, startHeightRef.current + delta)
      );
      setHeight(nextHeight);
    };

    const handleUp = () => {
      setIsResizing(false);
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [isResizing]);

  const handleScroll = React.useCallback((event: React.UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    const distanceToBottom = target.scrollHeight - (target.scrollTop + target.clientHeight);
    setAutoScroll(distanceToBottom < SCROLL_EPSILON);
  }, []);

  const handleManualRefresh = React.useCallback(() => {
    if (!workflowName) return;
    cancelPollingRef.current = false;
    pollLogs().catch(() => undefined);
  }, [pollLogs, workflowName]);

  const containerHeight = isOpen ? height : COLLAPSED_HEIGHT;

  return (
    <div
      className="relative shrink-0 border-t border-gray-800 bg-gray-950 text-gray-100 shadow-[0_-6px_16px_rgba(0,0,0,0.35)] transition-[height] duration-200"
      style={{ height: containerHeight }}
    >
      <div
        className="absolute inset-x-0 top-0 h-2 cursor-row-resize"
        onMouseDown={handleResizeStart}
        role="presentation"
      />
      <header className="flex h-11 items-center justify-between border-b border-gray-800 px-3 text-[12px] uppercase tracking-wide text-gray-300">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggle}
            className="flex items-center gap-2 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-gray-200 transition hover:bg-gray-800"
            aria-expanded={isOpen}
          >
            <span
              className={`inline-block transition-transform ${isOpen ? "rotate-0" : "-rotate-90"}`}
            >
              ▾
            </span>
            Terminal de ejecución
            {!isOpen && hasUnread && <span className="ml-1 inline-flex h-2 w-2 rounded-full bg-emerald-400" />}
          </button>
          {submitting && (
            <span className="text-[11px] font-medium text-amber-300">Enviando workflow…</span>
          )}
          {fetchError && (
            <span className="text-[11px] font-medium text-red-400">
              Error obteniendo logs
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-[11px] text-gray-400">
          {lastUpdated && (
            <span>Actualizado {formatTimestamp(lastUpdated)}</span>
          )}
          {loading && <span className="animate-pulse text-gray-200">Actualizando…</span>}
          <button
            type="button"
            onClick={handleManualRefresh}
            disabled={!workflowName || loading}
            className={`rounded border border-gray-700 px-2 py-1 text-[11px] uppercase tracking-wide transition ${
              !workflowName || loading
                ? "cursor-not-allowed text-gray-600"
                : "text-gray-200 hover:bg-gray-800"
            }`}
          >
            Actualizar
          </button>
        </div>
      </header>
      {isOpen && (
        <div className="flex h-[calc(100%-2.75rem)] flex-col">
          <div className="h-2 cursor-row-resize border-b border-gray-800" onMouseDown={handleResizeStart} />
          <div className="space-y-2 border-b border-gray-800 px-3 py-2 text-[11px] text-gray-300">
            {submitError && (
              <div className="rounded border border-red-500/40 bg-red-900/20 px-3 py-2 text-[11px] text-red-300">
                {submitError}
              </div>
            )}
            {submitResult && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-emerald-300">
                <span className="font-semibold text-emerald-200">
                  Workflow {submitResult.workflowName}
                </span>
                <span className="rounded border border-emerald-500/40 bg-emerald-900/20 px-2 py-0.5 text-[10px]">
                  Namespace {submitResult.namespace}
                </span>
                <span className="truncate text-[10px] text-emerald-400">
                  {submitResult.bucket}/{submitResult.key}
                </span>
                {submitResult.cliOutput && (
                  <span className="w-full truncate font-mono text-[10px] text-emerald-200">
                    {submitResult.cliOutput}
                  </span>
                )}
              </div>
            )}
            {!submitResult && !submitError && (
              <div className="text-[11px] text-gray-400">
                Envía un workflow para comenzar a ver los logs de ejecución.
              </div>
            )}
            {fetchError && (
              <div className="rounded border border-red-500/40 bg-red-900/20 px-3 py-2 text-[11px] text-red-300">
                {fetchError}
              </div>
            )}
          </div>
          <div className="flex-1 overflow-hidden">
            <div
              ref={scrollRef}
              className="h-full overflow-y-auto px-3 py-2 font-mono text-[12px] leading-5 text-gray-100"
              onScroll={handleScroll}
            >
              {lines.length === 0 && !loading && (
                <div className="py-6 text-center text-[11px] text-gray-500">
                  {submitResult
                    ? "No se han recibido logs todavía. Esta vista se actualizará automáticamente."
                    : "Sin ejecuciones. Envía un workflow para comenzar."}
                </div>
              )}
              {lines.map((line) => (
                <div key={line.id} className="flex items-start gap-3 py-1 text-[12px]">
                  <span className="w-16 shrink-0 text-[10px] text-gray-500">
                    {formatTimestamp(line.timestamp)}
                  </span>
                  <span
                    className="shrink-0 text-[11px] font-semibold"
                    style={{ color: line.color }}
                    title={line.slug}
                  >
                    {line.label}
                  </span>
                  <span className="shrink-0 max-w-[160px] truncate text-[10px] text-gray-500">
                    {line.podName}
                  </span>
                  <pre className="flex-1 whitespace-pre-wrap break-words">
                    {line.text || " "}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

