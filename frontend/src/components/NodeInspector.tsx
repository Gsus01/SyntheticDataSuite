"use client";

import React from "react";
import type { Edge, Node } from "reactflow";
import { API_BASE } from "@/lib/api";
import { NODE_TYPES } from "@/lib/flow-const";
import type { FlowNodeData, NodeArtifact } from "@/types/flow";
import { getOutputArtifacts, previewArtifact, type OutputArtifactInfo } from "@/lib/workflow";
import { derivePhase } from "@/lib/runtime-status";

type NodeInspectorProps = {
  isOpen: boolean;
  node: Node<FlowNodeData> | null;
  sessionId: string;
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  onChange: (nodeId: string, updater: (data: FlowNodeData) => FlowNodeData) => void;
};

type ValueKind = "number" | "boolean" | "string" | "array" | "object" | "null" | "unknown";

type ArtifactUploadResponsePayload = {
  bucket: string;
  key: string;
  size: number;
  content_type?: string | null;
  contentType?: string | null;
  original_filename?: string | null;
  originalFilename?: string | null;
};

type DownloadState = {
  loading: boolean;
  error: string | null;
};

type OutputPreviewState = {
  loading: boolean;
  error: string | null;
  content: string | null;
  truncated: boolean;
  contentType?: string | null;
  limitedLines?: boolean;
};

function extractFilenameFromDisposition(disposition: string | null): string | null {
  if (!disposition) return null;

  const filenameStarMatch = disposition.match(/filename\*=([^;]+)/i);
  if (filenameStarMatch && filenameStarMatch[1]) {
    const value = filenameStarMatch[1].trim().replace(/^"|"$/g, "");
    try {
      return decodeURIComponent(value.replace(/^UTF-8''/i, ""));
    } catch {
      return value;
    }
  }

  const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
  if (filenameMatch && filenameMatch[1]) {
    return filenameMatch[1].trim();
  }

  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function getValueKind(value: unknown): ValueKind {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  const valueType = typeof value;
  if (valueType === "number" || valueType === "boolean" || valueType === "string") return valueType;
  if (valueType === "object") return "object";
  return "unknown";
}

function formatValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === null || value === undefined) return "";
  return JSON.stringify(value, null, 2);
}

function formatBytes(size: number | null | undefined): string {
  if (!Number.isFinite(size) || size === null || size === undefined || size < 0) {
    return "-";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const decimals = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(decimals)} ${units[unitIndex]}`;
}

const PREVIEW_LINE_LIMIT = 100;

const PHASE_BADGE_MAP: Record<string, string> = {
  pending: "border-amber-200 bg-amber-50 text-amber-700",
  waiting: "border-amber-200 bg-amber-50 text-amber-700",
  queued: "border-amber-200 bg-amber-50 text-amber-700",
  running: "border-sky-200 bg-sky-50 text-sky-700",
  executing: "border-sky-200 bg-sky-50 text-sky-700",
  inprogress: "border-sky-200 bg-sky-50 text-sky-700",
  succeeded: "border-emerald-200 bg-emerald-50 text-emerald-700",
  completed: "border-emerald-200 bg-emerald-50 text-emerald-700",
  success: "border-emerald-200 bg-emerald-50 text-emerald-700",
  failed: "border-rose-200 bg-rose-50 text-rose-700",
  error: "border-rose-200 bg-rose-50 text-rose-700",
  terminated: "border-rose-200 bg-rose-50 text-rose-700",
  cancelled: "border-rose-200 bg-rose-50 text-rose-700",
  skipped: "border-gray-200 bg-gray-50 text-gray-600",
  omitted: "border-gray-200 bg-gray-50 text-gray-600",
};

const DEFAULT_PHASE_BADGE = "border-gray-200 bg-gray-50 text-gray-600";

function getPhaseBadgeClasses(phase?: string | null): string {
  if (!phase) return DEFAULT_PHASE_BADGE;
  const normalized = phase.trim().toLowerCase().replace(/\s+/g, "");
  return PHASE_BADGE_MAP[normalized] ?? DEFAULT_PHASE_BADGE;
}

function limitPreviewLines(content: string): { content: string; limitedLines: boolean } {
  const lines = content.split(/\r?\n/);
  if (lines.length <= PREVIEW_LINE_LIMIT) {
    return { content, limitedLines: false };
  }
  const limitedContent = lines.slice(0, PREVIEW_LINE_LIMIT).join("\n");
  return {
    content: limitedContent,
    limitedLines: true,
  };
}

function isEqualValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((value, index) => isEqualValue(value, b[index]));
  }

  if (isRecord(a) && isRecord(b)) {
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length) return false;
    return keysA.every((key) => isEqualValue(a[key], b[key]));
  }

  return Object.is(a, b);
}

export default function NodeInspector({
  isOpen,
  node,
  sessionId,
  nodes,
  edges,
  onChange,
}: NodeInspectorProps) {
  const [structuredInputs, setStructuredInputs] = React.useState<Record<string, string>>({});
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState<string | null>(null);
  const [outputArtifacts, setOutputArtifacts] = React.useState<OutputArtifactInfo[] | null>(null);
  const [outputLoading, setOutputLoading] = React.useState(false);
  const [outputError, setOutputError] = React.useState<string | null>(null);
  const [downloadState, setDownloadState] = React.useState<Record<string, DownloadState>>({});
  const [previewStates, setPreviewStates] = React.useState<Record<string, OutputPreviewState>>({});
  const safeTrim = (value?: string | null) => (typeof value === "string" ? value.trim() : "");
  const trimmedLabel = safeTrim(node?.data.label);
  const trimmedTemplate = safeTrim(node?.data.templateName);
  const displayLabel = trimmedLabel || trimmedTemplate || node?.id || "Nodo seleccionado";
  const argoWorkflowId = safeTrim(node?.data.runtimeStatus?.slug) || node?.id || "—";
  const componentId = node?.id || "—";
  const phaseLabel = safeTrim(derivePhase(node?.data.runtimeStatus));
  const phaseBadgeClasses = getPhaseBadgeClasses(phaseLabel);
  const showTemplateMeta =
    Boolean(trimmedTemplate) &&
    trimmedTemplate.toLowerCase() !== displayLabel.toLowerCase();

  React.useEffect(() => {
    if (!isOpen) return;
    setStructuredInputs({});
    setErrors({});
    setUploadError(null);
    setUploadSuccess(null);
    setUploading(false);
    setOutputArtifacts(null);
    setOutputLoading(false);
    setOutputError(null);
    setDownloadState({});
    setPreviewStates({});
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [isOpen, node?.id]);

  React.useEffect(() => {
    if (!isOpen || !node || node.type !== NODE_TYPES.nodeOutput) {
      setOutputArtifacts(null);
      setOutputLoading(false);
      setOutputError(null);
      setDownloadState({});
      setPreviewStates({});
      return;
    }

    let cancelled = false;
    async function loadArtifacts() {
      setOutputLoading(true);
      setOutputError(null);
      try {
        const artifacts = await getOutputArtifacts(
          {
            sessionId,
            nodes,
            edges,
          },
          node.id
        );
        if (!cancelled) {
          setOutputArtifacts(artifacts);
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : "Error obteniendo artefactos";
          setOutputError(message);
          setOutputArtifacts([]);
        }
      } finally {
        if (!cancelled) {
          setOutputLoading(false);
        }
      }
    }

    loadArtifacts();

    return () => {
      cancelled = true;
    };
  }, [edges, isOpen, node, node?.id, node?.type, nodes, sessionId]);

  const clearParameter = React.useCallback(
    (key: string) => {
      if (!node) return;
      onChange(node.id, (data) => {
        const existing = isRecord(data.parameters) ? { ...data.parameters } : {};
        if (key in existing) {
          delete existing[key];
        }
        const hasOverrides = Object.keys(existing).length > 0;
        return {
          ...data,
          parameters: hasOverrides ? existing : undefined,
        };
      });
      setStructuredInputs((prev) => {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      });
      setErrors((prev) => {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      });
    },
    [node, onChange]
  );

  const updateParameter = React.useCallback(
    (key: string, value: unknown) => {
      if (!node) return;
      onChange(node.id, (data) => {
        const existing = isRecord(data.parameters) ? { ...data.parameters } : {};
        const defaults = isRecord(data.parameterDefaults) ? data.parameterDefaults : {};
        const defaultValue = defaults[key];

        if (defaultValue !== undefined && isEqualValue(value, defaultValue)) {
          if (key in existing) {
            delete existing[key];
          }
        } else {
          existing[key] = value;
        }

        const hasOverrides = Object.keys(existing).length > 0;
        return {
          ...data,
          parameters: hasOverrides ? existing : undefined,
        };
      });
    },
    [node, onChange]
  );

  const parameters = React.useMemo(() => {
    const raw = node?.data?.parameters;
    return isRecord(raw) ? raw : {};
  }, [node?.data?.parameters]);

  const defaults = React.useMemo(() => {
    const raw = node?.data?.parameterDefaults;
    return isRecord(raw) ? raw : {};
  }, [node?.data?.parameterDefaults]);

  const declaredKeys = React.useMemo(() => {
    const raw = node?.data?.parameterKeys;
    if (!Array.isArray(raw)) return [];
    return raw.filter((key): key is string => typeof key === "string");
  }, [node?.data?.parameterKeys]);

  const rawKeys = React.useMemo(() => {
    if (declaredKeys.length) return declaredKeys;
    return Array.from(new Set([...Object.keys(defaults), ...Object.keys(parameters)]));
  }, [declaredKeys, defaults, parameters]);

  const keys = React.useMemo(
    () => rawKeys.filter((key) => key !== "bucket" && key !== "key"),
    [rawKeys]
  );

  const nodeIdentifier = React.useMemo(() => {
    if (!node) return null;
    const templateName = node.data?.templateName;
    if (typeof templateName === "string" && templateName.trim()) {
      return templateName.trim();
    }
    return node.id;
  }, [node]);

  const handleFileUpload = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      if (!node || !nodeIdentifier) return;

      const selectedFile = event.target.files?.[0];
      if (!selectedFile) return;

      setUploading(true);
      setUploadError(null);
      setUploadSuccess(null);

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("session_id", sessionId);
      formData.append("node_id", nodeIdentifier);
      formData.append("artifact_name", selectedFile.name);

      try {
        const response = await fetch(`${API_BASE}/artifacts/upload`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          let detail = `HTTP ${response.status}`;
          try {
            const errorPayload = await response.json();
            if (errorPayload?.detail) {
              detail = typeof errorPayload.detail === "string"
                ? errorPayload.detail
                : JSON.stringify(errorPayload.detail);
            }
          } catch {
            // ignore JSON parse errors
          }
          throw new Error(detail);
        }

        const payload = (await response.json()) as ArtifactUploadResponsePayload;
        const normalized: NodeArtifact = {
          bucket: payload.bucket,
          key: payload.key,
          size: payload.size,
          contentType: payload.contentType ?? payload.content_type ?? null,
          originalFilename:
            payload.originalFilename ?? payload.original_filename ?? selectedFile.name ?? null,
        };

        onChange(node.id, (data) => {
          const existing = isRecord(data.parameters) ? { ...data.parameters } : {};
          existing.bucket = normalized.bucket;
          existing.key = normalized.key;
          return {
            ...data,
            parameters: existing,
            uploadedArtifact: normalized,
          };
        });

        setUploadSuccess("Archivo subido correctamente");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Error subiendo archivo";
        setUploadError(message);
      } finally {
        setUploading(false);
        event.target.value = "";
      }
    },
    [node, nodeIdentifier, onChange, sessionId]
  );

  const uploadedArtifact = node?.data.uploadedArtifact;
  const hasBucketKeyParams = rawKeys.includes("bucket") && rawKeys.includes("key");
  const isOutputNode = node?.type === NODE_TYPES.nodeOutput;
  const shouldShowUpload = Boolean(node && !isOutputNode && (node.type === NODE_TYPES.nodeInput || hasBucketKeyParams));

  const handleUploadCardClick = React.useCallback(() => {
    if (uploading) return;
    fileInputRef.current?.click();
  }, [uploading]);

  const handleUploadCardKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        handleUploadCardClick();
      }
    },
    [handleUploadCardClick]
  );

  const handleDownloadArtifact = React.useCallback(async (artifact: OutputArtifactInfo) => {
    const key = artifact.key;
    setDownloadState((prev) => ({
      ...prev,
      [key]: {
        loading: true,
        error: null,
      },
    }));

    let errorMessage: string | null = null;

    try {
      const url = new URL(`${API_BASE}/artifacts/download`);
      url.searchParams.set("bucket", artifact.bucket);
      url.searchParams.set("key", artifact.key);

      const response = await fetch(url.toString());
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const text = await response.text();
          if (text) detail = text;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition");
      const fallback = artifact.sourceArtifactName || artifact.key.split("/").pop() || "artifact";
      const filename = extractFilenameFromDisposition(disposition) || fallback;

      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(objectUrl);
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Error descargando artefacto";
    }

    setDownloadState((prev) => ({
      ...prev,
      [key]: {
        loading: false,
        error: errorMessage,
      },
    }));
  }, []);

  const handleTogglePreview = React.useCallback(
    async (artifact: OutputArtifactInfo) => {
      const key = artifact.key;
      const current = previewStates[key];

      if (current && current.content && !current.loading && !current.error) {
        setPreviewStates((prev) => {
          const next = { ...prev };
          delete next[key];
          return next;
        });
        return;
      }

      setPreviewStates((prev) => ({
        ...prev,
        [key]: {
          loading: true,
          error: null,
          content: current?.content ?? null,
          truncated: current?.truncated ?? false,
          contentType: current?.contentType,
          limitedLines: current?.limitedLines ?? false,
        },
      }));

      try {
        const result = await previewArtifact(artifact.bucket, artifact.key);
        const { content: limitedContent, limitedLines } = limitPreviewLines(result.content);
        setPreviewStates((prev) => ({
          ...prev,
          [key]: {
            loading: false,
            error: null,
            content: limitedContent,
            truncated: result.truncated || limitedLines,
            contentType: result.contentType ?? undefined,
            limitedLines,
          },
        }));
      } catch (error) {
        const message = error instanceof Error ? error.message : "Error obteniendo previsualización";
        setPreviewStates((prev) => ({
          ...prev,
          [key]: {
            loading: false,
            error: message,
            content: null,
            truncated: false,
            contentType: undefined,
            limitedLines: false,
          },
        }));
      }
    },
    [previewStates]
  );

  if (!isOpen) {
    return null;
  }

  const renderStructuredInput = (key: string, value: unknown) => {
    const current = structuredInputs[key] ?? formatValue(value);
    const error = errors[key];

    return (
      <div className="flex flex-col gap-1">
        <textarea
          className="min-h-[120px] w-full resize-y rounded border border-gray-300 bg-white p-2 font-mono text-xs text-gray-800 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          value={current}
          onChange={(event) =>
            setStructuredInputs((prev) => ({
              ...prev,
              [key]: event.target.value,
            }))
          }
          onBlur={() => {
            const raw = structuredInputs[key];
            const trimmed = raw?.trim();
            if (!trimmed) {
              clearParameter(key);
              setStructuredInputs((prev) => {
                const next = { ...prev };
                delete next[key];
                return next;
              });
              setErrors((prev) => {
                const next = { ...prev };
                delete next[key];
                return next;
              });
              return;
            }

            try {
              const parsed = JSON.parse(trimmed);
              updateParameter(key, parsed);
              setStructuredInputs((prev) => {
                const next = { ...prev };
                delete next[key];
                return next;
              });
              setErrors((prev) => {
                const next = { ...prev };
                delete next[key];
                return next;
              });
            } catch {
              setErrors((prev) => ({
                ...prev,
                [key]: "JSON inválido. Revisa el formato.",
              }));
            }
          }}
          placeholder="Introduce JSON válido"
        />
        {error ? (
          <span className="text-xs text-red-600">{error}</span>
        ) : (
          <span className="text-xs text-gray-400">Edición como JSON</span>
        )}
      </div>
    );
  };

  const renderControl = (key: string, activeValue: unknown, defaultValue: unknown) => {
    const kind = getValueKind(defaultValue ?? activeValue);

    if (kind === "number") {
      const value = typeof activeValue === "number" ? activeValue : "";
      return (
        <input
          type="number"
          className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-800 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          value={value}
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === "") {
              clearParameter(key);
              return;
            }
            const parsed = Number(raw);
            if (!Number.isNaN(parsed)) updateParameter(key, parsed);
          }}
        />
      );
    }

    if (kind === "boolean") {
      const value = typeof activeValue === "boolean" ? String(activeValue) : "default";
      return (
        <select
          className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-800 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          value={value}
          onChange={(event) => {
            const next = event.target.value;
            if (next === "default") {
              clearParameter(key);
            } else {
              updateParameter(key, next === "true");
            }
          }}
        >
          <option value="default">Predeterminado</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      );
    }

    if (kind === "string" || kind === "null" || kind === "unknown") {
      const value = typeof activeValue === "string" ? activeValue : "";
      return (
        <input
          type="text"
          className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-sm text-gray-800 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          value={value}
          onChange={(event) => updateParameter(key, event.target.value)}
          onBlur={(event) => {
            if (event.target.value === "") {
              clearParameter(key);
            }
          }}
        />
      );
    }

    return renderStructuredInput(key, activeValue);
  };

  const handleReset = (key: string) => {
    if (!node) return;
    onChange(node.id, (data) => {
      const existing = isRecord(data.parameters) ? { ...data.parameters } : {};
      if (key in existing) {
        delete existing[key];
      }
      const hasOverrides = Object.keys(existing).length > 0;
      return {
        ...data,
        parameters: hasOverrides ? existing : undefined,
      };
    });
    setStructuredInputs((prev) => {
      if (!(key in prev)) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
    setErrors((prev) => {
      if (!(key in prev)) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  return (
    <aside className="relative z-20 flex h-full min-h-0 w-80 shrink-0 flex-col overflow-hidden border-l border-gray-200 bg-white text-sm text-gray-800 shadow-xl">
      <div className="border-b border-gray-200 px-4 pb-4 pt-4">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Inspector</div>
        {node ? (
          <>
            <div className="mt-1 text-base font-semibold text-gray-900">{displayLabel}</div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-wide text-gray-500">
              <span className="rounded-full border border-gray-200 bg-gray-50 px-2 py-0.5 font-semibold text-gray-600">
                Nodo <span className="font-mono text-[10px] uppercase tracking-normal">{componentId}</span>
              </span>
              {argoWorkflowId && argoWorkflowId !== "—" && (
                <span className="rounded-full border border-gray-200 bg-gray-50 px-2 py-0.5 font-semibold text-gray-600">
                  Argo <span className="font-mono text-[10px] uppercase tracking-normal">{argoWorkflowId}</span>
                </span>
              )}
              {phaseLabel && (
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${phaseBadgeClasses}`}
                >
                  {phaseLabel}
                </span>
              )}
              {showTemplateMeta && (
                <span className="rounded-full border border-indigo-200 bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold text-indigo-600">
                  {trimmedTemplate}
                </span>
              )}
            </div>
          </>
        ) : (
          <div className="mt-1 text-sm text-gray-500">Selecciona un nodo.</div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {!node ? (
          <div className="text-xs text-gray-500">Selecciona un nodo en el lienzo para editar sus parámetros.</div>
        ) : (
          <div className="flex flex-col gap-4 pb-8">
            {isOutputNode && (
              <div className="rounded border border-indigo-200 bg-indigo-50 p-3 text-xs text-gray-700">
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-indigo-500">
                  Artefactos de salida
                </div>
              {outputLoading && <span className="text-[11px] text-gray-500">Buscando artefactos…</span>}
              {outputError && <span className="text-[11px] text-red-600">{outputError}</span>}
              {!outputLoading && !outputError && (!outputArtifacts || outputArtifacts.length === 0) && (
                <span className="text-[11px] text-gray-500">
                  Conecta este nodo a la salida de otro para ver los artefactos generados.
                </span>
              )}
              {outputArtifacts?.map((artifact) => {
                const downloadStatus = downloadState[artifact.key] ?? { loading: false, error: null };
                const previewState = previewStates[artifact.key];
                return (
                  <div
                    key={artifact.key}
                    className="mt-2 rounded border border-indigo-100 bg-white p-2 text-[11px] text-gray-700 shadow-sm"
                  >
                    <div className="text-xs font-semibold text-gray-600">
                      {artifact.sourceArtifactName || artifact.inputName}
                    </div>
                    <div className="mt-1 space-y-1 text-[11px] text-gray-600">
                      <div>
                        <span className="font-semibold text-gray-500">Bucket:</span> {artifact.bucket}
                      </div>
                      <div className="break-all">
                        <span className="font-semibold text-gray-500">Key:</span> {artifact.key}
                      </div>
                      {artifact.size !== undefined && artifact.size !== null && (
                        <div>
                          <span className="font-semibold text-gray-500">Tamaño:</span> {formatBytes(artifact.size)}
                        </div>
                      )}
                      {artifact.contentType && (
                        <div>
                          <span className="font-semibold text-gray-500">Content-Type:</span> {artifact.contentType}
                        </div>
                      )}
                      {artifact.sourceNodeId && (
                        <div>
                          <span className="font-semibold text-gray-500">Origen:</span> {artifact.sourceNodeId}
                        </div>
                      )}
                      {!artifact.exists && (
                        <div className="text-amber-600">
                          El archivo aún no está disponible en MinIO. Ejecuta el workflow y vuelve a intentarlo.
                        </div>
                      )}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => handleDownloadArtifact(artifact)}
                        disabled={downloadStatus.loading}
                        className={`rounded border px-2 py-1 text-[11px] uppercase tracking-wide shadow-sm transition ${
                          downloadStatus.loading
                            ? "cursor-wait border-indigo-200 bg-indigo-100 text-indigo-500"
                            : "border-indigo-300 text-indigo-600 hover:border-indigo-400 hover:bg-indigo-50"
                        }`}
                      >
                        {downloadStatus.loading ? "Descargando…" : "Descargar"}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleTogglePreview(artifact)}
                        disabled={previewState?.loading}
                        className={`rounded border px-2 py-1 text-[11px] uppercase tracking-wide shadow-sm transition ${
                          previewState?.loading
                            ? "cursor-wait border-indigo-200 bg-indigo-100 text-indigo-500"
                            : "border-indigo-300 text-indigo-600 hover:border-indigo-400 hover:bg-indigo-50"
                        }`}
                      >
                        {previewState?.loading
                          ? "Cargando…"
                          : previewState?.content && !previewState.error
                          ? "Ocultar"
                          : "Previsualizar"}
                      </button>
                    </div>
                    {downloadStatus.error && (
                      <div className="mt-1 text-[11px] text-red-600">{downloadStatus.error}</div>
                    )}
                    {previewState?.error && (
                      <div className="mt-1 text-[11px] text-red-600">{previewState.error}</div>
                    )}
                    {previewState?.content && !previewState.loading && !previewState.error && (
                      <div className="mt-2 max-h-48 overflow-auto rounded bg-gray-900 p-2 font-mono text-[11px] leading-relaxed text-gray-100">
                        <pre className="whitespace-pre-wrap break-words text-gray-100">
                          {previewState.content}
                        </pre>
                        {previewState?.limitedLines && (
                          <span className="mt-1 block text-[10px] text-gray-400">
                            Se muestran las primeras {PREVIEW_LINE_LIMIT} líneas del artefacto.
                          </span>
                        )}
                        {previewState.truncated && (
                          <span className="mt-1 block text-[10px] text-amber-300">
                            Contenido truncado. Descarga el archivo para verlo completo.
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {shouldShowUpload && (
            <div
              className={`rounded border border-gray-200 bg-gray-50 p-3 text-xs text-gray-700 transition-colors ${
                uploading ? "cursor-default" : "cursor-pointer hover:border-indigo-300 hover:bg-indigo-50"
              }`}
              role="button"
              tabIndex={0}
              onClick={handleUploadCardClick}
              onKeyDown={handleUploadCardKeyDown}
            >
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                disabled={uploading}
                onChange={handleFileUpload}
              />
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-gray-500">
                Archivo de entrada
              </div>
              <div className="flex flex-col gap-2">
                {uploading && <span className="text-xs text-gray-500">Subiendo…</span>}
                {uploadError && <span className="text-xs text-red-600">{uploadError}</span>}
                {uploadSuccess && !uploadError && (
                  <span className="text-xs text-emerald-600">{uploadSuccess}</span>
                )}
                {uploadedArtifact ? (
                  <div className="mt-1 space-y-1 text-[11px] text-gray-600">
                    <div>
                      <span className="font-semibold text-gray-500">Último fichero:</span> {uploadedArtifact.originalFilename || "(sin nombre)"}
                    </div>
                    <div>
                      <span className="font-semibold text-gray-500">Bucket:</span> {uploadedArtifact.bucket}
                    </div>
                    <div>
                      <span className="font-semibold text-gray-500">Key:</span> {uploadedArtifact.key}
                    </div>
                    <div>
                      <span className="font-semibold text-gray-500">Tamaño:</span> {formatBytes(uploadedArtifact.size)}
                    </div>
                  </div>
                ) : (
                  <span className="text-[11px] text-gray-500">Selecciona un fichero para subirlo a MinIO.</span>
                )}
              </div>
            </div>
          )}

          {keys.length === 0 ? (
            <div className="text-xs text-gray-500">Este nodo no expone parámetros configurables.</div>
          ) : (
            keys.map((key) => {
              const defaultValue = defaults[key];
              const isCustom = key in parameters;
              const activeValue = isCustom ? parameters[key] : defaultValue;
              const showDefaultBadge = !isCustom && defaultValue !== undefined;
              const showCustomBadge = isCustom;

              return (
                <div key={key} className="flex flex-col gap-1">
                  <label className="flex items-center text-xs font-semibold uppercase tracking-wide text-gray-500">
                    <span>{key}</span>
                    {showDefaultBadge && (
                      <span className="ml-2 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-gray-500">
                        Por defecto
                      </span>
                    )}
                    {showCustomBadge && (
                      <span className="ml-2 rounded bg-indigo-100 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-indigo-600">
                        Personalizado
                      </span>
                    )}
                  </label>
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      {renderControl(key, activeValue, defaultValue)}
                    </div>
                    <button
                      type="button"
                      onClick={() => handleReset(key)}
                      disabled={!isCustom}
                      className={`rounded border px-2 py-1 text-[11px] uppercase tracking-wide shadow-sm transition ${
                        isCustom
                          ? "border-gray-300 text-gray-600 hover:border-gray-400 hover:text-gray-800"
                          : "cursor-not-allowed border-gray-200 bg-gray-100 text-gray-300"
                      }`}
                    >
                      Reset
                    </button>
                  </div>
                  {defaultValue !== undefined && (
                    <span className="text-[11px] text-gray-400">
                      Predeterminado: {formatValue(defaultValue)}
                    </span>
                  )}
                </div>
              );
            })
          )}
          </div>
        )}
      </div>
    </aside>
  );
}
