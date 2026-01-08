"use client";

import React from "react";
import NodeCard from "@/components/NodeCard";
import { DND_MIME, NODE_TYPES, NODE_META_MIME, type NodeTypeId } from "@/lib/flow-const";
import {
  computeConnectablePorts,
  fetchNodeTemplates,
  inferNodeType,
  type CatalogNodeTemplate,
} from "@/lib/node-templates";

function inferRfType(tpl: CatalogNodeTemplate): NodeTypeId {
  return inferNodeType(tpl);
}

function onDragStartCatalog(event: React.DragEvent<HTMLDivElement>, tpl: CatalogNodeTemplate) {
  const rfType = inferRfType(tpl);
  event.dataTransfer.setData(DND_MIME, rfType);
  event.dataTransfer.setData("text/plain", tpl.name);
  const tone = (tpl.type || "other").toLowerCase();
  const parameterKeys = tpl.parameter_defaults
    ? Object.keys(tpl.parameter_defaults)
    : [];
  const artifactPorts = computeConnectablePorts(tpl.artifacts);
  event.dataTransfer.setData(
    NODE_META_MIME,
    JSON.stringify({
      tone,
      templateName: tpl.name,
      parameterDefaults: tpl.parameter_defaults ?? null,
      parameterKeys,
      artifactPorts,
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
  const [templates, setTemplates] = React.useState<CatalogNodeTemplate[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchNodeTemplates();
        if (!cancelled) {
          setTemplates(data);
        }
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
    const groups: Record<string, CatalogNodeTemplate[]> = {
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
