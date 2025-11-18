"use client";

import React from "react";
import type { Node } from "reactflow";

import {
  fetchWorkflowLogs,
  type SubmitWorkflowResult,
} from "@/lib/workflow";
import type { FlowNodeData } from "@/types/flow";

type WorkflowTerminalProps = {
  isOpen: boolean;
  onToggle: () => void;
  workflowName?: string | null;
  namespace?: string | null;
  nodeSlugMap?: Record<string, string> | null;
  nodes: Node<FlowNodeData>[];
  submitting: boolean;
  submitResult: SubmitWorkflowResult | null;
  submitError: string | null;
};

const MIN_HEIGHT = 180;
const MAX_HEIGHT = 560;
const COLLAPSED_HEIGHT = 44;

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function formatTimestamp(timestampMs: number): string {
  return timeFormatter.format(new Date(timestampMs));
}

const SCROLL_EPSILON = 32;
const MAX_RENDERED_LOG_LINES = 2000;
const NODE_FALLBACK_LABEL = "workflow";

type LogLevel = "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" | "FATAL" | "LOG";

type ParsedLogEntry = {
  id: string;
  lineNumber: number;
  nodeLabel: string;
  nodeColor: string;
  levelLabel: LogLevel;
  content: string;
  timestamp?: string;
};

const nodeColorCache = new Map<string, string>();

const LEVEL_CLASSNAMES: Record<LogLevel, string> = {
  TRACE: "border-indigo-400/60 text-indigo-200 bg-indigo-500/10",
  DEBUG: "border-sky-400/60 text-sky-200 bg-sky-500/10",
  INFO: "border-emerald-400/60 text-emerald-200 bg-emerald-500/10",
  WARN: "border-amber-500/60 text-amber-200 bg-amber-500/10",
  ERROR: "border-rose-500/70 text-rose-200 bg-rose-500/10",
  FATAL: "border-red-500/80 text-red-100 bg-red-500/20",
  LOG: "border-gray-600 text-gray-300 bg-gray-800/40",
};

const LEVEL_TOKENS = ["TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL"];

function stripAnsiCodes(value: string): string {
  return value.replace(/\u001b\[[0-9;]*m/g, "");
}

function normalizeHintKey(value: string): string {
  return value.trim().toLowerCase();
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^0-9a-z]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function resolveDisplayLabel(slug?: string | null, node?: Node<FlowNodeData>): string {
  const explicitLabel = node?.data.label?.trim();
  if (explicitLabel) {
    return explicitLabel;
  }
  const templateName = node?.data.templateName?.trim();
  if (templateName) {
    return templateName;
  }
  if (slug && slug.trim()) {
    return humanizeSlug(slug.trim()) || slug.trim();
  }
  const fallbackId = node?.id?.trim();
  if (fallbackId) {
    return humanizeSlug(fallbackId) || fallbackId;
  }
  return NODE_FALLBACK_LABEL;
}

function buildNodeLabelHints(
  nodeSlugMap?: Record<string, string> | null,
  nodes?: Node<FlowNodeData>[]
): Map<string, string> {
  const hints = new Map<string, string>();
  const nodeById = new Map<string, Node<FlowNodeData>>();
  (nodes || []).forEach((node) => {
    nodeById.set(node.id, node);
  });

  const register = (value?: string | null, label?: string | null) => {
    if (!value || !label) return;
    const normalized = normalizeHintKey(value);
    if (normalized && !hints.has(normalized)) {
      hints.set(normalized, label);
    }
  };

  const addHint = (raw?: string | null, label?: string | null) => {
    if (!raw || !label) return;
    register(raw, label);
    register(slugify(raw), label);
    raw
      .split(/[^0-9A-Za-z]+/)
      .filter(Boolean)
      .forEach((token) => register(token, label));
  };

  if (nodeSlugMap) {
    Object.entries(nodeSlugMap).forEach(([nodeId, slug]) => {
      const node = nodeById.get(nodeId);
      const displayLabel = resolveDisplayLabel(slug, node);
      addHint(slug, displayLabel);
      addHint(nodeId, displayLabel);
      addHint(node?.data.label, displayLabel);
      addHint(node?.data.templateName, displayLabel);
    });
  }

  (nodes || []).forEach((node) => {
    const displayLabel = resolveDisplayLabel(nodeSlugMap?.[node.id], node);
    addHint(node.id, displayLabel);
    addHint(node.data.label, displayLabel);
    addHint(node.data.templateName, displayLabel);
    const slugged = slugify(displayLabel);
    if (slugged) {
      addHint(slugged, displayLabel);
    }
  });

  return hints;
}

function humanizeSlug(value: string): string {
  return value
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function colorForNode(label: string): string {
  const cached = nodeColorCache.get(label);
  if (cached) return cached;
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = (hash * 33 + label.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  const color = `hsl(${hue}, 60%, 62%)`;
  nodeColorCache.set(label, color);
  return color;
}

function normalizeLevelToken(token: string): LogLevel {
  const upper = token.trim().toUpperCase();
  if (upper === "WARNING") return "WARN";
  if (LEVEL_TOKENS.includes(upper)) {
    return (upper === "WARNING" ? "WARN" : (upper as LogLevel));
  }
  return "LOG";
}

function isLikelyLevelToken(token: string): boolean {
  return normalizeLevelToken(token) !== "LOG";
}

function extractNodeSegment(line: string): { nodeLabel: string; remainder: string } {
  const trimmed = line.trimStart();

  const colonMatch = trimmed.match(/^[ \t]*([A-Za-z][A-Za-z0-9._-]{1,})(?::|\s+-)\s*(.*)$/);
  if (colonMatch) {
    return {
      nodeLabel: colonMatch[1],
      remainder: colonMatch[2] ?? "",
    };
  }

  if (trimmed.startsWith("[")) {
    const bracket = trimmed.match(/^\[([^\]]+)\]\s*(.*)$/);
    if (bracket && !isLikelyLevelToken(bracket[1])) {
      return {
        nodeLabel: bracket[1],
        remainder: bracket[2] ?? "",
      };
    }
  }

  return {
    nodeLabel: NODE_FALLBACK_LABEL,
    remainder: trimmed,
  };
}

function deriveNodeLabel(rawLabel: string, hints: Map<string, string>): string {
  const sanitized = rawLabel
    .replace(/[^0-9A-Za-z._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "");

  const tokens = sanitized.split("-").filter(Boolean);

  const findHint = (token: string) => {
    const normalized = normalizeHintKey(token);
    return hints.get(normalized);
  };

  for (let i = 0; i < tokens.length; i += 1) {
    const candidate = tokens.slice(i).join("-");
    const hint = findHint(candidate);
    if (hint) {
      return hint;
    }
    const trimmedDigits = candidate.replace(/(^\d+)|(\d+$)/g, "");
    if (trimmedDigits) {
      const trimmedHint = findHint(trimmedDigits);
      if (trimmedHint) {
        return trimmedHint;
      }
    }
  }

  for (let i = tokens.length - 1; i >= 0; i -= 1) {
    const token = tokens[i];
    if (/^[0-9]+$/.test(token) || /^[0-9a-f]{6,}$/i.test(token)) {
      continue;
    }
    const hint = findHint(token) || findHint(token.replace(/^\d+|\d+$/g, ""));
    if (hint) {
      return hint;
    }
  }

  const slugJoined = tokens.join("-");
  const hintFull = findHint(slugJoined);
  if (hintFull) {
    return hintFull;
  }

  for (let i = tokens.length - 1; i >= 0; i -= 1) {
    const candidate = tokens[i];
    if (!candidate) continue;
    if (/^[0-9]+$/.test(candidate)) continue;
    if (/^[0-9a-f]{6,}$/i.test(candidate)) continue;
    return humanizeSlug(candidate) || candidate;
  }

  return NODE_FALLBACK_LABEL;
}

function extractTimestampSegment(text: string): { timestamp?: string; remainder: string } {
  const match = text.match(
    /^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)(?:\s*(?:-|–|—)\s*|\s+)(.*)$/
  );
  if (match) {
    return {
      timestamp: match[1],
      remainder: match[2] ?? "",
    };
  }

  const timeKv = text.match(/\btime="?([0-9T:.\-+Z]+)"?/i);
  if (timeKv) {
    const idx = timeKv.index ?? 0;
    const end = idx + timeKv[0].length;
    const before = text.slice(0, idx).trimEnd();
    const after = text.slice(end).trimStart();
    const remainder = [before, after].filter(Boolean).join(" ").trim();
    return {
      timestamp: timeKv[1],
      remainder: remainder || "",
    };
  }

  return { remainder: text };
}

function formatInlineTimestamp(raw?: string): string | undefined {
  if (!raw) return undefined;
  const normalized = raw.replace(",", ".").replace(" ", "T");
  const parsed = Date.parse(normalized);
  if (!Number.isNaN(parsed)) {
    const date = new Date(parsed);
    const hh = String(date.getHours()).padStart(2, "0");
    const mm = String(date.getMinutes()).padStart(2, "0");
    const ss = String(date.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }
  const fallback = raw.match(/(\d{2}:\d{2}:\d{2})/);
  return fallback ? fallback[1] : raw;
}

function extractLevelSegment(text: string): { levelLabel: LogLevel; remainder: string } {
  const working = text.trimStart();

  const bracket = working.match(/^\[([^\]]+)\]\s*(.*)$/);
  if (bracket && isLikelyLevelToken(bracket[1])) {
    return {
      levelLabel: normalizeLevelToken(bracket[1]),
      remainder: bracket[2] ?? "",
    };
  }

  const prefix = working.match(/^(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b[:\-]?\s*(.*)$/i);
  if (prefix) {
    return {
      levelLabel: normalizeLevelToken(prefix[1]),
      remainder: prefix[2] ?? "",
    };
  }

  const inline = working.match(
    /^(.*?)(?:\s+-\s+|\s+)(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s*-\s*(.*)$/i
  );
  if (inline) {
    const prefixText = inline[1].trim();
    const suffixText = inline[3].trimStart();
    const remainder = prefixText && suffixText ? `${prefixText} ${suffixText}` : prefixText || suffixText;
    return {
      levelLabel: normalizeLevelToken(inline[2]),
      remainder: remainder ?? "",
    };
  }

  const kv = working.match(/\b(level|lvl)\s*[:=]\s*(trace|debug|info|warn|warning|error|fatal)\b/i);
  if (kv) {
    const idx = kv.index ?? 0;
    const before = working.slice(0, idx).trimEnd();
    const after = working.slice(idx + kv[0].length).trimStart();
    const remainder = `${before}${before && after ? " " : ""}${after}`.trimStart();
    return {
      levelLabel: normalizeLevelToken(kv[2]),
      remainder,
    };
  }

  return {
    levelLabel: "LOG",
    remainder: working,
  };
}

function simplifyKeyValueContent(text: string): string {
  const hasKeyValue = /\b[A-Za-z_][\w-]*=/.test(text);
  if (!hasKeyValue) {
    return text;
  }

  let message: string | null = null;
  const extras: string[] = [];
  const regex = /\b([A-Za-z_][\w-]*)=("[^"]*"|<[^>]+>|[^\s"]+)/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    const key = match[1].toLowerCase();
    let value = match[2];
    if (value.startsWith('"') && value.endsWith('"')) {
      value = value.slice(1, -1);
    }
    if (key === "time") {
      continue;
    }
    if (key === "msg") {
      message = value;
    } else {
      extras.push(`${match[1]}=${value}`);
    }
  }

  if (!message && !extras.length) {
    return text.trim();
  }

  const base = (message ?? "").trim();
  const extraText = extras.length ? `(${extras.join(" ")})` : "";
  return [base, extraText].filter(Boolean).join(" ").trim();
}

function buildLogEntries(raw: string, labelHints: Map<string, string>): ParsedLogEntry[] {
  if (!raw) return [];
  const lines = raw.split(/\r?\n/);
  const total = lines.length;
  const start = Math.max(0, total - MAX_RENDERED_LOG_LINES);
  const sliced = lines.slice(start);
  const assignments = new Map<string, string>();
  const baseCounts = new Map<string, number>();

  const assignLabel = (rawKey: string, baseLabel: string) => {
    const normalizedRaw = normalizeHintKey(rawKey || baseLabel);
    const existing = assignments.get(normalizedRaw);
    if (existing) {
      return existing;
    }
    const nextCount = (baseCounts.get(baseLabel) ?? 0) + 1;
    baseCounts.set(baseLabel, nextCount);
    const label = nextCount === 1 ? baseLabel : `${baseLabel}-${nextCount}`;
    assignments.set(normalizedRaw, label);
    return label;
  };

  return sliced
    .map((line, idx) => {
      const cleanLine = stripAnsiCodes(line);
      const lineNumber = start + idx + 1;
      const { nodeLabel: rawNodeLabel, remainder } = extractNodeSegment(cleanLine);
      const baseNodeLabel = deriveNodeLabel(rawNodeLabel, labelHints);
      const displayNodeLabel = assignLabel(rawNodeLabel, baseNodeLabel);
      const { timestamp, remainder: afterTimestamp } = extractTimestampSegment(remainder);
      const { levelLabel, remainder: finalContent } = extractLevelSegment(afterTimestamp);
      const simplified = simplifyKeyValueContent(finalContent);
      let contentText = simplified || finalContent.trim();
      if (!contentText) {
        contentText = afterTimestamp.trim();
      }
      if (!contentText) {
        contentText = cleanLine.trim();
      }
      return {
        id: `${lineNumber}:${displayNodeLabel}:${idx}`,
        lineNumber,
        nodeLabel: displayNodeLabel,
        nodeColor: colorForNode(baseNodeLabel),
        levelLabel,
        content: contentText || "",
        timestamp: formatInlineTimestamp(timestamp),
      };
    })
    .filter((entry, idx) => !(sliced[idx] === "" && idx === sliced.length - 1));
}

export default function WorkflowTerminal({
  isOpen,
  onToggle,
  workflowName,
  namespace,
  nodeSlugMap,
  nodes,
  submitting,
  submitResult,
  submitError,
}: WorkflowTerminalProps) {
  const resolvedNamespace = namespace ?? submitResult?.namespace ?? undefined;
  const [height, setHeight] = React.useState<number>(320);
  const [isResizing, setIsResizing] = React.useState(false);
  const [fetchError, setFetchError] = React.useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = React.useState<number | null>(null);
  const [hasUnread, setHasUnread] = React.useState(false);
  const [autoScroll, setAutoScroll] = React.useState(true);
  const [logText, setLogText] = React.useState<string>("");

  const scheduleRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelPollingRef = React.useRef(false);
  const isFetchingRef = React.useRef(false);
  const scrollRef = React.useRef<HTMLDivElement | null>(null);
  const startYRef = React.useRef(0);
  const startHeightRef = React.useRef(0);

  const labelHints = React.useMemo(() => buildNodeLabelHints(nodeSlugMap, nodes), [nodeSlugMap, nodes]);
  const logEntries = React.useMemo(() => buildLogEntries(logText, labelHints), [labelHints, logText]);
  const lineCount = logEntries.length;

  const pollLogs = React.useCallback(async () => {
    if (!workflowName || isFetchingRef.current) {
      return;
    }

    if (scheduleRef.current) {
      clearTimeout(scheduleRef.current);
      scheduleRef.current = null;
    }

    isFetchingRef.current = true;
    try {
      const result = await fetchWorkflowLogs({
        workflowName,
        namespace: resolvedNamespace,
      });

      const nextLogs = result.logs ?? "";
      let changed = false;
      setLogText((prev) => {
        if (prev === nextLogs) {
          return prev;
        }
        changed = true;
        return nextLogs;
      });

      if (changed) {
        setLastUpdated(Date.now());
        if (!isOpen) {
          setHasUnread(true);
        }
      }

      setFetchError(null);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Error obteniendo logs de la ejecución";
      setFetchError(message);
    } finally {
      isFetchingRef.current = false;

      if (!cancelPollingRef.current && workflowName) {
        scheduleRef.current = setTimeout(() => {
          pollLogs().catch(() => undefined);
        }, 4000);
      }
    }
  }, [isOpen, resolvedNamespace, workflowName]);

  React.useEffect(() => {
    setLogText("");
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
  }, [autoScroll, logText]);

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
      className="relative shrink-0 border-t border-gray-800 bg-gray-950 text-gray-100 shadow-[0_-6px_16px_rgba(0,0,0,0.35)]"
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
            className="flex items-center gap-2 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-gray-200 transition hover:bg-gray-800 cursor-pointer"
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
          <span className="text-[10px] text-gray-500">
            Líneas: {lineCount}
          </span>
          <button
            type="button"
            onClick={handleManualRefresh}
            disabled={!workflowName}
            className={`rounded border border-gray-700 px-2 py-1 text-[11px] uppercase tracking-wide transition ${
              !workflowName
                ? "cursor-not-allowed text-gray-600"
                : "text-gray-200 hover:bg-gray-800 cursor-pointer"
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
              {logEntries.length === 0 ? (
                <div className="py-6 text-center text-[11px] text-gray-500">
                  {submitResult
                    ? "No se han recibido logs todavía. Esta vista se actualizará automáticamente."
                    : "Sin ejecuciones. Envía un workflow para comenzar."}
                </div>
              ) : (
                logEntries.map((entry) => (
                  <div key={entry.id} className="flex items-start gap-3 py-1 text-[12px]">
                    <span className="w-16 shrink-0 text-right font-mono text-[10px] text-gray-500">
                      {entry.timestamp ?? entry.lineNumber.toString().padStart(4, "0")}
                    </span>
                    <span
                      className="shrink-0 rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
                      style={{
                        color: entry.nodeColor,
                        borderColor: entry.nodeColor,
                      }}
                    >
                      {entry.nodeLabel}
                    </span>
                    <span
                      className={`shrink-0 rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${LEVEL_CLASSNAMES[entry.levelLabel]}`}
                    >
                      {entry.levelLabel}
                    </span>
                    <pre className="flex-1 whitespace-pre-wrap break-words text-gray-100">
                      {entry.content || " "}
                    </pre>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
