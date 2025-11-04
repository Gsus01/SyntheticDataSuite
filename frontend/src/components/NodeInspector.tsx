"use client";

import React from "react";
import type { Node } from "reactflow";
import type { FlowNodeData } from "@/types/flow";

type NodeInspectorProps = {
  isOpen: boolean;
  node: Node<FlowNodeData> | null;
  onChange: (nodeId: string, updater: (data: FlowNodeData) => FlowNodeData) => void;
};

type ValueKind = "number" | "boolean" | "string" | "array" | "object" | "null" | "unknown";

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

export default function NodeInspector({ isOpen, node, onChange }: NodeInspectorProps) {
  const [structuredInputs, setStructuredInputs] = React.useState<Record<string, string>>({});
  const [errors, setErrors] = React.useState<Record<string, string>>({});

  React.useEffect(() => {
    if (!isOpen) return;
    setStructuredInputs({});
    setErrors({});
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

  if (!isOpen) {
    return null;
  }

  const parameters = isRecord(node?.data.parameters) ? node!.data.parameters : {};
  const defaults = isRecord(node?.data.parameterDefaults) ? node!.data.parameterDefaults : {};
  const declaredKeys = Array.isArray(node?.data.parameterKeys)
    ? node!.data.parameterKeys.filter((key): key is string => typeof key === "string")
    : [];
  const keys = declaredKeys.length
    ? declaredKeys
    : Array.from(new Set([...Object.keys(defaults), ...Object.keys(parameters)]));

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
      ) : keys.length === 0 ? (
        <div className="text-xs text-gray-500">Este nodo no expone parámetros configurables.</div>
      ) : (
        <div className="flex flex-col gap-4">
          {keys.map((key) => {
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
          })}
        </div>
      )}
    </aside>
  );
}

