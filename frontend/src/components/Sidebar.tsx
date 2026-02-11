"use client";

import React from "react";
import NodeCard from "@/components/NodeCard";
import { DND_MIME, NODE_TYPES, NODE_META_MIME, type NodeTypeId } from "@/lib/flow-const";
import {
  activateCatalogComponentVersion,
  computeConnectablePorts,
  deleteCatalogComponent,
  fetchNodeTemplates,
  getCatalogComponentVersions,
  inferNodeType,
  listCatalogComponents,
  registerCatalogComponent,
  type CatalogComponentVersion,
  type CatalogNodeTemplate,
} from "@/lib/node-templates";

const UNDO_WINDOW_MS = 10_000;

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
      <div className="mb-2 mt-3 text-[11px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
        {title}
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

type SidebarProps = {
  onTemplatesDeleted?: (templateNames: string[]) => void;
  onTemplatesRestored?: (templateNames: string[]) => void;
};

type TemplateCardProps = {
  template: CatalogNodeTemplate;
  tone: "input" | "output" | "preprocessing" | "training" | "generation" | "other";
  managing: boolean;
  selected: boolean;
  disabled: boolean;
  onToggleSelection: (templateName: string) => void;
};

type DeletedComponentSnapshot = {
  name: string;
  activeVersion: string | null;
  versions: CatalogComponentVersion[];
};

type UndoToastState = {
  deletedNames: string[];
  snapshots: DeletedComponentSnapshot[];
};

function TemplateCard({
  template,
  tone,
  managing,
  selected,
  disabled,
  onToggleSelection,
}: TemplateCardProps) {
  const variant =
    inferRfType(template) === NODE_TYPES.nodeInput
      ? "input"
      : inferRfType(template) === NODE_TYPES.nodeOutput
        ? "output"
        : "default";

  const handleToggle = () => {
    if (disabled) return;
    onToggleSelection(template.name);
  };

  const nodeCardContent = (
    <div
      role="button"
      tabIndex={0}
      draggable={!managing && !disabled}
      onDragStart={(event) => {
        if (!managing && !disabled) {
          onDragStartCatalog(event, template);
        }
      }}
      className={`min-w-0 flex-1 ${managing ? "cursor-pointer" : "cursor-grab active:cursor-grabbing"}`}
    >
      <NodeCard label={template.name} variant={variant} tone={tone} compact />
    </div>
  );

  if (!managing) {
    return nodeCardContent;
  }

  return (
    <div
      className={`flex items-center gap-2 rounded border p-1.5 transition ${selected
        ? "border-indigo-300 bg-indigo-50 dark:border-indigo-700 dark:bg-indigo-950/30"
        : "border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800"
        }`}
      onClick={managing ? handleToggle : undefined}
    >
      {managing && (
        <input
          type="checkbox"
          checked={selected}
          onChange={handleToggle}
          onClick={(event) => event.stopPropagation()}
          disabled={disabled}
          className="h-4 w-4 cursor-pointer rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 disabled:cursor-not-allowed"
          aria-label={`Seleccionar ${template.name}`}
        />
      )}
      {nodeCardContent}
    </div>
  );
}

export default function Sidebar({ onTemplatesDeleted, onTemplatesRestored }: SidebarProps) {
  const [templates, setTemplates] = React.useState<CatalogNodeTemplate[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [deleteError, setDeleteError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [manageMode, setManageMode] = React.useState(false);
  const [selectedTemplateNames, setSelectedTemplateNames] = React.useState<string[]>([]);
  const [showDeleteConfirm, setShowDeleteConfirm] = React.useState(false);
  const [deletingSelected, setDeletingSelected] = React.useState(false);
  const [undoingDelete, setUndoingDelete] = React.useState(false);
  const [undoToast, setUndoToast] = React.useState<UndoToastState | null>(null);
  const undoTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearUndoTimer = React.useCallback(() => {
    if (undoTimerRef.current) {
      clearTimeout(undoTimerRef.current);
      undoTimerRef.current = null;
    }
  }, []);

  const scheduleUndoExpiry = React.useCallback(() => {
    clearUndoTimer();
    undoTimerRef.current = setTimeout(() => {
      setUndoToast(null);
    }, UNDO_WINDOW_MS);
  }, [clearUndoTimer]);

  React.useEffect(() => {
    return () => {
      clearUndoTimer();
    };
  }, [clearUndoTimer]);

  const loadTemplates = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchNodeTemplates();
      setTemplates(data);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "Error cargando catálogo";
      setError(message || "Error cargando catálogo");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void loadTemplates();
  }, [loadTemplates]);

  React.useEffect(() => {
    if (!templates) return;
    const validNames = new Set(templates.map((template) => template.name));
    setSelectedTemplateNames((prev) => prev.filter((name) => validNames.has(name)));
  }, [templates]);

  const byType = React.useMemo(() => {
    const groups: Record<string, CatalogNodeTemplate[]> = {
      input: [],
      output: [],
      preprocessing: [],
      training: [],
      generation: [],
      other: [],
    };
    (templates || []).forEach((template) => {
      const key = (template.type || "other").toLowerCase();
      if (key in groups) {
        groups[key].push(template);
      } else {
        groups.other.push(template);
      }
    });
    return groups;
  }, [templates]);

  const selectedCount = selectedTemplateNames.length;
  const busy = deletingSelected || undoingDelete;

  const toggleManageMode = React.useCallback(() => {
    setDeleteError(null);
    setManageMode((prev) => {
      if (prev) {
        setSelectedTemplateNames([]);
        setShowDeleteConfirm(false);
        return false;
      }
      return true;
    });
  }, []);

  const toggleTemplateSelection = React.useCallback((templateName: string) => {
    setSelectedTemplateNames((prev) =>
      prev.includes(templateName)
        ? prev.filter((name) => name !== templateName)
        : [...prev, templateName]
    );
  }, []);

  const openBulkDeleteConfirm = React.useCallback(() => {
    if (selectedCount === 0 || busy) {
      return;
    }
    setDeleteError(null);
    setShowDeleteConfirm(true);
  }, [busy, selectedCount]);

  const closeBulkDeleteConfirm = React.useCallback(() => {
    if (!busy) {
      setShowDeleteConfirm(false);
    }
  }, [busy]);

  const handleDeleteSelected = React.useCallback(async () => {
    if (!selectedTemplateNames.length) {
      return;
    }

    const namesToDelete = [...selectedTemplateNames];
    setDeletingSelected(true);
    setDeleteError(null);

    try {
      const [componentList, versionSnapshots] = await Promise.all([
        listCatalogComponents(),
        Promise.all(
          namesToDelete.map(async (name) => {
            const versions = await getCatalogComponentVersions(name);
            return { name, versions };
          })
        ),
      ]);

      const activeVersionMap = new Map(
        componentList.map((item) => [item.name, item.activeVersion ?? null])
      );

      await Promise.all(namesToDelete.map((name) => deleteCatalogComponent(name)));

      const snapshots: DeletedComponentSnapshot[] = versionSnapshots.map((snapshot) => ({
        name: snapshot.name,
        activeVersion: activeVersionMap.get(snapshot.name) ?? null,
        versions: snapshot.versions,
      }));

      setTemplates((prev) =>
        prev ? prev.filter((template) => !namesToDelete.includes(template.name)) : prev
      );
      onTemplatesDeleted?.(namesToDelete);

      setUndoToast({
        deletedNames: namesToDelete,
        snapshots,
      });
      scheduleUndoExpiry();

      setShowDeleteConfirm(false);
      setManageMode(false);
      setSelectedTemplateNames([]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudieron eliminar los elementos seleccionados.";
      setDeleteError(message);
    } finally {
      setDeletingSelected(false);
    }
  }, [onTemplatesDeleted, scheduleUndoExpiry, selectedTemplateNames]);

  const dismissUndoToast = React.useCallback(() => {
    clearUndoTimer();
    setUndoToast(null);
  }, [clearUndoTimer]);

  const handleUndoDelete = React.useCallback(async () => {
    if (!undoToast) {
      return;
    }

    setUndoingDelete(true);
    setDeleteError(null);

    try {
      for (const snapshot of undoToast.snapshots) {
        const orderedVersions = [...snapshot.versions].sort((a, b) => {
          const aTime = a.createdAt ? Date.parse(a.createdAt) : 0;
          const bTime = b.createdAt ? Date.parse(b.createdAt) : 0;
          return aTime - bTime;
        });

        for (const versionInfo of orderedVersions) {
          await registerCatalogComponent(versionInfo.spec, false);
        }

        if (snapshot.activeVersion) {
          await activateCatalogComponentVersion(snapshot.name, snapshot.activeVersion);
        }
      }

      await loadTemplates();
      onTemplatesRestored?.(undoToast.deletedNames);
      dismissUndoToast();
    } catch (err) {
      const message = err instanceof Error ? err.message : "No se pudo deshacer la eliminación.";
      setDeleteError(message);
    } finally {
      setUndoingDelete(false);
    }
  }, [dismissUndoToast, loadTemplates, onTemplatesRestored, undoToast]);

  return (
    <>
      <aside className="flex h-full min-h-0 w-64 shrink-0 flex-col border-r border-gray-200 bg-gray-50 p-3 text-sm text-gray-800 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-100">
        <div className="mb-3 flex items-center justify-between gap-2">
          <div className="font-medium text-gray-600 uppercase tracking-wide dark:text-gray-400">Nodos</div>
          <button
            type="button"
            onClick={toggleManageMode}
            disabled={busy}
            className="cursor-pointer rounded border border-gray-300 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-gray-700 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
          >
            {manageMode ? "Cancelar" : "Gestionar"}
          </button>
        </div>

        {manageMode && (
          <div className="mb-3 rounded border border-rose-200 bg-rose-50 p-2 dark:border-rose-900/40 dark:bg-rose-950/20">
            <div className="mb-2 text-[11px] font-medium text-rose-700 dark:text-rose-300">
              {selectedCount} seleccionado(s)
            </div>
            <button
              type="button"
              onClick={openBulkDeleteConfirm}
              disabled={selectedCount === 0 || busy}
              className="w-full cursor-pointer rounded bg-rose-600 px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-white shadow-sm transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Eliminar seleccionados ({selectedCount})
            </button>
          </div>
        )}

        <div className="flex flex-1 min-h-0 flex-col overflow-y-auto pr-1">
          <div className="mb-2 mt-1 text-[11px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
            Catálogo
          </div>

          {loading && <div className="text-xs text-gray-500 dark:text-gray-400">Cargando catálogo…</div>}

          {error && (
            <div className="text-xs text-red-600 dark:text-red-400">No se pudo cargar: {error}</div>
          )}

          {deleteError && (
            <div className="mt-2 text-xs text-red-600 dark:text-red-400">{deleteError}</div>
          )}

          {!loading && !error && templates && (
            <div>
              {byType.input.length > 0 && (
                <Section title="Entrada">
                  {byType.input.map((template) => (
                    <TemplateCard
                      key={template.name}
                      template={template}
                      tone="input"
                      managing={manageMode}
                      selected={selectedTemplateNames.includes(template.name)}
                      disabled={busy}
                      onToggleSelection={toggleTemplateSelection}
                    />
                  ))}
                </Section>
              )}
              {byType.output.length > 0 && (
                <Section title="Salida">
                  {byType.output.map((template) => (
                    <TemplateCard
                      key={template.name}
                      template={template}
                      tone="output"
                      managing={manageMode}
                      selected={selectedTemplateNames.includes(template.name)}
                      disabled={busy}
                      onToggleSelection={toggleTemplateSelection}
                    />
                  ))}
                </Section>
              )}
              {byType.preprocessing.length > 0 && (
                <Section title="Preprocesamiento">
                  {byType.preprocessing.map((template) => (
                    <TemplateCard
                      key={template.name}
                      template={template}
                      tone="preprocessing"
                      managing={manageMode}
                      selected={selectedTemplateNames.includes(template.name)}
                      disabled={busy}
                      onToggleSelection={toggleTemplateSelection}
                    />
                  ))}
                </Section>
              )}
              {byType.training.length > 0 && (
                <Section title="Entrenamiento">
                  {byType.training.map((template) => (
                    <TemplateCard
                      key={template.name}
                      template={template}
                      tone="training"
                      managing={manageMode}
                      selected={selectedTemplateNames.includes(template.name)}
                      disabled={busy}
                      onToggleSelection={toggleTemplateSelection}
                    />
                  ))}
                </Section>
              )}
              {byType.generation.length > 0 && (
                <Section title="Generación">
                  {byType.generation.map((template) => (
                    <TemplateCard
                      key={template.name}
                      template={template}
                      tone="generation"
                      managing={manageMode}
                      selected={selectedTemplateNames.includes(template.name)}
                      disabled={busy}
                      onToggleSelection={toggleTemplateSelection}
                    />
                  ))}
                </Section>
              )}
              {byType.other.length > 0 && (
                <Section title="Otros">
                  {byType.other.map((template) => (
                    <TemplateCard
                      key={template.name}
                      template={template}
                      tone="other"
                      managing={manageMode}
                      selected={selectedTemplateNames.includes(template.name)}
                      disabled={busy}
                      onToggleSelection={toggleTemplateSelection}
                    />
                  ))}
                </Section>
              )}
            </div>
          )}
        </div>
      </aside>

      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4 backdrop-blur-sm dark:bg-black/60">
          <div className="w-full max-w-sm rounded-lg bg-white p-6 shadow-xl ring-1 ring-gray-900/5 dark:bg-gray-900 dark:ring-gray-800/60">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">¿Eliminar selección del catálogo?</h3>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              Se eliminarán <span className="font-semibold">{selectedCount}</span> elementos y se quitarán del editor.
            </p>
            {deleteError && <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">{deleteError}</p>}
            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={closeBulkDeleteConfirm}
                disabled={busy}
                className="cursor-pointer rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700 dark:focus:ring-offset-gray-900"
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={handleDeleteSelected}
                disabled={busy}
                className="cursor-pointer rounded-md bg-rose-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-rose-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70 dark:focus:ring-offset-gray-900"
              >
                {deletingSelected ? "Eliminando…" : `Eliminar seleccionados (${selectedCount})`}
              </button>
            </div>
          </div>
        </div>
      )}

      {undoToast && (
        <div className="fixed bottom-4 left-4 z-50 w-full max-w-sm rounded-lg border border-emerald-200 bg-white p-4 shadow-xl dark:border-emerald-900/50 dark:bg-gray-900">
          <p className="text-sm text-gray-700 dark:text-gray-200">
            Se eliminaron <span className="font-semibold">{undoToast.deletedNames.length}</span> elementos del catálogo.
          </p>
          <div className="mt-3 flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={dismissUndoToast}
              disabled={undoingDelete}
              className="cursor-pointer rounded border border-gray-300 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
            >
              Cerrar
            </button>
            <button
              type="button"
              onClick={handleUndoDelete}
              disabled={undoingDelete}
              className="cursor-pointer rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {undoingDelete ? "Restaurando…" : "Deshacer"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
