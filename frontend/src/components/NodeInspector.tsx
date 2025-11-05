"use client";

import React from "react";
import type { Node } from "reactflow";
import { API_BASE } from "@/lib/api";
import { NODE_TYPES } from "@/lib/flow-const";
import type { FlowNodeData, NodeArtifact } from "@/types/flow";

type NodeInspectorProps = {
  isOpen: boolean;
  node: Node<FlowNodeData> | null;
  sessionId: string;
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

export default function NodeInspector({ isOpen, node, sessionId, onChange }: NodeInspectorProps) {
  const [structuredInputs, setStructuredInputs] = React.useState<Record<string, string>>({});
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!isOpen) return;
    setStructuredInputs({});
    setErrors({});
    setUploadError(null);
    setUploadSuccess(null);
    setUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [isOpen, node?.id]);

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
    [defaults, node, nodeIdentifier, onChange, parameters, sessionId]
  );

  const uploadedArtifact = node?.data.uploadedArtifact;
  const hasBucketKeyParams = rawKeys.includes("bucket") && rawKeys.includes("key");
  const shouldShowUpload = Boolean(node && (node.type === NODE_TYPES.nodeInput || hasBucketKeyParams));

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
    <aside className="relative w-80 shrink-0 border-l border-gray-200 bg-white p-4 text-sm text-gray-800 shadow-xl">
      <div className="mb-4">
        <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Inspector</div>
        {node ? (
          <div className="mt-1 text-sm font-medium text-gray-900">{node.data.label || node.id}</div>
        ) : (
          <div className="mt-1 text-sm text-gray-500">Selecciona un nodo.</div>
        )}
        {node?.data.templateName && (
          <div className="mt-1 text-xs text-gray-500">Plantilla: {node.data.templateName}</div>
        )}
      </div>

      {!node ? (
        <div className="text-xs text-gray-500">Selecciona un nodo en el lienzo para editar sus parámetros.</div>
      ) : (
        <div className="flex flex-col gap-4">
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
    </aside>
  );
}

