"use client";

import React from "react";
import NodeCard from "@/components/NodeCard";
import { API_BASE } from "@/lib/api";
import { DND_MIME, NODE_TYPES, NODE_META_MIME, type NodeTypeId } from "@/lib/flow-const";

type Item = {
  label: string;
  type: NodeTypeId;
};

const baseItems: Item[] = [
  { label: "Input Node", type: NODE_TYPES.nodeInput },
  { label: "Default Node", type: NODE_TYPES.nodeDefault },
  { label: "Output Node", type: NODE_TYPES.nodeOutput },
];

type ApiArtifact = { name: string; path?: string };
type ApiNodeTemplate = {
  name: string;
  type: string; // preprocessing | training | generation | ...
  parameters: string[];
  artifacts: { inputs: ApiArtifact[]; outputs: ApiArtifact[] };
  limits?: Record<string, unknown>;
  version?: string;
  parameter_defaults?: Record<string, unknown>;
};

function onDragStart(event: React.DragEvent<HTMLDivElement>, nodeType: Item["type"], label?: string) {
  event.dataTransfer.setData(DND_MIME, nodeType);
  if (label) event.dataTransfer.setData("text/plain", label);
  event.dataTransfer.setData(NODE_META_MIME, JSON.stringify({ tone: "other" }));
  event.dataTransfer.effectAllowed = "move";
}

function inferRfType(tpl: ApiNodeTemplate): NodeTypeId {
  const inputs = tpl.artifacts?.inputs?.length ?? 0;
  const outputs = tpl.artifacts?.outputs?.length ?? 0;
  if (inputs === 0 && outputs > 0) return NODE_TYPES.nodeInput; // source-only
  if (inputs > 0 && outputs === 0) return NODE_TYPES.nodeOutput; // target-only
  return NODE_TYPES.nodeDefault; // both or none
}

function onDragStartCatalog(event: React.DragEvent<HTMLDivElement>, tpl: ApiNodeTemplate) {
  const rfType = inferRfType(tpl);
  event.dataTransfer.setData(DND_MIME, rfType);
  event.dataTransfer.setData("text/plain", tpl.name);
  const tone = (tpl.type || "other").toLowerCase();
  const parameterKeys = tpl.parameter_defaults
    ? Object.keys(tpl.parameter_defaults)
    : [];
  event.dataTransfer.setData(
    NODE_META_MIME,
    JSON.stringify({
      tone,
      templateName: tpl.name,
      parameterDefaults: tpl.parameter_defaults ?? null,
      parameterKeys,
    })
  );
  event.dataTransfer.effectAllowed = "move";
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <div className="mb-2 mt-3 text-[11px] font-semibold uppercase tracking-wider text-gray-500">
        {title}
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

export default function Sidebar() {
  const [templates, setTemplates] = React.useState<ApiNodeTemplate[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/workflow-templates`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: ApiNodeTemplate[] = await res.json();
        if (!cancelled) setTemplates(data);
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : "Error cargando catálogo";
          setError(message || "Error cargando catálogo");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const byType = React.useMemo(() => {
    const groups: Record<string, ApiNodeTemplate[]> = {
      input: [],
      output: [],
      preprocessing: [],
      training: [],
      generation: [],
      other: [],
    };
    (templates || []).forEach((t) => {
      const key = (t.type || "other").toLowerCase();
      if (key in groups) groups[key].push(t);
      else groups.other.push(t);
    });
    return groups;
  }, [templates]);

  return (
    <aside className="flex h-full min-h-0 w-64 shrink-0 flex-col border-r border-gray-200 bg-gray-50 p-3 text-sm text-gray-800">
      <div className="mb-3 font-medium text-gray-600 uppercase tracking-wide">Nodos</div>
      <div className="flex flex-1 min-h-0 flex-col overflow-y-auto pr-1">
        {/* Base nodes for prototyping */}
        <Section title="Básicos">
          {baseItems.map((item) => (
            <div
              key={item.type}
              role="button"
              tabIndex={0}
              draggable
              onDragStart={(e) => onDragStart(e, item.type, item.label)}
              className="cursor-grab active:cursor-grabbing"
            >
              <NodeCard
                label={item.label}
                variant={
                  item.type === "nodeInput"
                    ? "input"
                    : item.type === "nodeDefault"
                    ? "default"
                    : "output"
                }
                compact
              />
            </div>
          ))}
        </Section>

        {/* Catalog from backend */}
        <div className="mb-2 mt-4 text-[11px] font-semibold uppercase tracking-wider text-gray-500">
          Catálogo
        </div>
        {loading && <div className="text-xs text-gray-500">Cargando catálogo…</div>}
        {error && (
          <div className="text-xs text-red-600">No se pudo cargar: {error}</div>
        )}
        {!loading && !error && templates && (
          <div>
            {byType.input.length > 0 && (
              <Section title="Entrada">
                {byType.input.map((t) => (
                  <div
                    key={t.name}
                    role="button"
                    tabIndex={0}
                    draggable
                    onDragStart={(e) => onDragStartCatalog(e, t)}
                    className="cursor-grab active:cursor-grabbing"
                  >
                    <NodeCard label={t.name} variant={inferRfType(t) === NODE_TYPES.nodeInput ? "input" : inferRfType(t) === NODE_TYPES.nodeOutput ? "output" : "default"} tone="input" compact />
                  </div>
                ))}
              </Section>
            )}
            {byType.output.length > 0 && (
              <Section title="Salida">
                {byType.output.map((t) => (
                  <div
                    key={t.name}
                    role="button"
                    tabIndex={0}
                    draggable
                    onDragStart={(e) => onDragStartCatalog(e, t)}
                    className="cursor-grab active:cursor-grabbing"
                  >
                    <NodeCard
                      label={t.name}
                      variant={inferRfType(t) === NODE_TYPES.nodeOutput ? "output" : inferRfType(t) === NODE_TYPES.nodeInput ? "input" : "default"}
                      tone="output"
                      compact
                    />
                  </div>
                ))}
              </Section>
            )}
            {byType.preprocessing.length > 0 && (
              <Section title="Preprocesamiento">
                {byType.preprocessing.map((t) => (
                  <div
                    key={t.name}
                    role="button"
                    tabIndex={0}
                    draggable
                    onDragStart={(e) => onDragStartCatalog(e, t)}
                    className="cursor-grab active:cursor-grabbing"
                  >
                    <NodeCard label={t.name} variant={inferRfType(t) === NODE_TYPES.nodeInput ? "input" : inferRfType(t) === NODE_TYPES.nodeOutput ? "output" : "default"} tone="preprocessing" compact />
                  </div>
                ))}
              </Section>
            )}
            {byType.training.length > 0 && (
              <Section title="Entrenamiento">
                {byType.training.map((t) => (
                  <div
                    key={t.name}
                    role="button"
                    tabIndex={0}
                    draggable
                    onDragStart={(e) => onDragStartCatalog(e, t)}
                    className="cursor-grab active:cursor-grabbing"
                  >
                    <NodeCard label={t.name} variant={inferRfType(t) === NODE_TYPES.nodeInput ? "input" : inferRfType(t) === NODE_TYPES.nodeOutput ? "output" : "default"} tone="training" compact />
                  </div>
                ))}
              </Section>
            )}
            {byType.generation.length > 0 && (
              <Section title="Generación">
                {byType.generation.map((t) => (
                  <div
                    key={t.name}
                    role="button"
                    tabIndex={0}
                    draggable
                    onDragStart={(e) => onDragStartCatalog(e, t)}
                    className="cursor-grab active:cursor-grabbing"
                  >
                    <NodeCard label={t.name} variant={inferRfType(t) === NODE_TYPES.nodeInput ? "input" : inferRfType(t) === NODE_TYPES.nodeOutput ? "output" : "default"} tone="generation" compact />
                  </div>
                ))}
              </Section>
            )}
            {byType.other.length > 0 && (
              <Section title="Otros">
                {byType.other.map((t) => (
                  <div
                    key={t.name}
                    role="button"
                    tabIndex={0}
                    draggable
                    onDragStart={(e) => onDragStartCatalog(e, t)}
                    className="cursor-grab active:cursor-grabbing"
                  >
                    <NodeCard label={t.name} variant={inferRfType(t) === NODE_TYPES.nodeInput ? "input" : inferRfType(t) === NODE_TYPES.nodeOutput ? "output" : "default"} tone="other" compact />
                  </div>
                ))}
              </Section>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
