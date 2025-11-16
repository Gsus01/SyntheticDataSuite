"use client";

import React, { useCallback, useMemo, useRef } from "react";
import ReactFlow, {
  addEdge,
  Connection,
  Edge,
  Node,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import Sidebar from "@/components/Sidebar";
import { DefaultNode, InputNode, OutputNode } from "@/components/nodes/StyledNodes";
import NodeInspector from "@/components/NodeInspector";
import WorkflowTerminal from "@/components/WorkflowTerminal";
import { DND_MIME, NODE_TYPES, NODE_META_MIME, type NodeTypeId } from "@/lib/flow-const";
import type { FlowNodeData, WorkflowNodeRuntimeStatus } from "@/types/flow";
import {
  submitWorkflow,
  fetchWorkflowStatus,
  compileWorkflow,
  saveWorkflow,
  listWorkflows,
  getWorkflow,
  type SubmitWorkflowResult,
  type WorkflowStatusNodeMap,
  type WorkflowRecord,
  type WorkflowSummary,
} from "@/lib/workflow";
import { normalizeStatusForDisplay } from "@/lib/runtime-status";

const INITIAL_PENDING_PHASE = "Pending";

function createPendingStatus(identifier: string, previous?: WorkflowNodeRuntimeStatus): WorkflowNodeRuntimeStatus {
  return {
    slug: previous?.slug ?? identifier,
    displayName: previous?.displayName,
    phase: INITIAL_PENDING_PHASE,
    message: undefined,
    progress: undefined,
    startedAt: undefined,
    finishedAt: undefined,
  };
}

const STATUS_POLL_INTERVAL_FAST = 600;
const STATUS_POLL_INTERVAL_WAITING = 450;
const STATUS_POLL_INTERVAL_ERROR = 4000;

let id = 0;
const getId = () => `dnd_${id++}`;

type DragMetaPayload = {
  tone?: unknown;
  templateName?: unknown;
  parameterKeys?: unknown;
  parameterDefaults?: unknown;
};

function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `sess_${Math.random().toString(36).slice(2, 10)}`;
}

type CompiledWorkflowState = {
  manifest: string;
  manifestFilename: string;
  nodeSlugMap: Record<string, string>;
  bucket?: string;
  compiledAt: string;
};

type SaveFormState = {
  name: string;
  description: string;
};

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatDateLabel(value?: string | null): string {
  if (!value) return "—";
  try {
    return dateFormatter.format(new Date(value));
  } catch {
    return value;
  }
}

function sanitizeNodesForSave(nodes: Node<FlowNodeData>[]): unknown[] {
  return nodes.map((node) => {
    const dataCopy = JSON.parse(JSON.stringify(node.data)) as FlowNodeData;
    delete (dataCopy as Record<string, unknown>).runtimeStatus;
    return {
      ...node,
      data: dataCopy,
    };
  });
}

function sanitizeEdgesForSave(edges: Edge[]): unknown[] {
  return edges.map((edge) => ({
    ...edge,
  }));
}

function EditorInner() {
  const reactFlowWrapper = useRef<HTMLDivElement | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  const { project } = useReactFlow();
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(null);
  const [sessionId, setSessionId] = React.useState(() => generateSessionId());
  const [submitting, setSubmitting] = React.useState(false);
  const [submitError, setSubmitError] = React.useState<string | null>(null);
  const [submitResult, setSubmitResult] = React.useState<SubmitWorkflowResult | null>(null);
  const [terminalOpen, setTerminalOpen] = React.useState(false);
  const [compiling, setCompiling] = React.useState(false);
  const [compileError, setCompileError] = React.useState<string | null>(null);
  const [compiledState, setCompiledState] = React.useState<CompiledWorkflowState | null>(null);
  const [isCompileDirty, setIsCompileDirty] = React.useState(true);
  const [hasUnsavedChanges, setHasUnsavedChanges] = React.useState(true);
  const [activeWorkflow, setActiveWorkflow] = React.useState<WorkflowRecord | null>(null);
  const [showSaveModal, setShowSaveModal] = React.useState(false);
  const [showLoadModal, setShowLoadModal] = React.useState(false);
  const [saveForm, setSaveForm] = React.useState<SaveFormState>({ name: "", description: "" });
  const [saving, setSaving] = React.useState(false);
  const [saveError, setSaveError] = React.useState<string | null>(null);
  const [saveNotice, setSaveNotice] = React.useState<string | null>(null);
  const [pendingOverwriteName, setPendingOverwriteName] = React.useState<string | null>(null);
  const [workflowSummaries, setWorkflowSummaries] = React.useState<WorkflowSummary[]>([]);
  const [summariesLoading, setSummariesLoading] = React.useState(false);
  const [summariesError, setSummariesError] = React.useState<string | null>(null);
  const [loadingWorkflowId, setLoadingWorkflowId] = React.useState<string | null>(null);
  const statusPollTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const statusAbortRef = React.useRef(false);
  const isReadyToSend = Boolean(compiledState && !isCompileDirty);
  const showUnsyncedHint = Boolean(compiledState && isCompileDirty);

  const markDirty = useCallback(
    (scope: "all" | "compile" | "layout" = "all") => {
      if (scope === "all" || scope === "compile") {
        setIsCompileDirty(true);
      }
      if (scope === "all" || scope === "layout") {
        setHasUnsavedChanges(true);
      }
    },
    []
  );

  React.useEffect(() => {
    setPendingOverwriteName(null);
    setSaveNotice(null);
  }, [saveForm.name]);

  const cancelStatusPolling = React.useCallback(() => {
    statusAbortRef.current = true;
    if (statusPollTimeoutRef.current) {
      clearTimeout(statusPollTimeoutRef.current);
      statusPollTimeoutRef.current = null;
    }
  }, []);

  const resetNodeRuntimeStatus = useCallback(() => {
    setNodes((prev) =>
      prev.map((node) => {
        const shouldTrack = node.type === NODE_TYPES.nodeDefault;
        if (!shouldTrack) {
          if (!node.data.runtimeStatus) {
            return node;
          }
          const nextData = { ...node.data };
          delete (nextData as Record<string, unknown>).runtimeStatus;
          return {
            ...node,
            data: nextData,
          };
        }
        return {
          ...node,
          data: {
            ...node.data,
            runtimeStatus: createPendingStatus(node.data.runtimeStatus?.slug ?? node.id, node.data.runtimeStatus),
          },
        };
      })
    );
  }, [setNodes]);

  const ensureInitialRuntimeStatus = useCallback(() => {
    setNodes((prev) =>
      prev.map((node) => {
        const shouldTrack = node.type === NODE_TYPES.nodeDefault;
        if (!shouldTrack) {
          if (!node.data.runtimeStatus) {
            return node;
          }
          const nextData = { ...node.data };
          delete (nextData as Record<string, unknown>).runtimeStatus;
          return {
            ...node,
            data: nextData,
          };
        }
        if (node.data.runtimeStatus) {
          return node;
        }
        return {
          ...node,
          data: {
            ...node.data,
            runtimeStatus: createPendingStatus(node.id),
          },
        };
      })
    );
  }, [setNodes]);

  React.useEffect(() => cancelStatusPolling, [cancelStatusPolling]);
  React.useEffect(() => {
    ensureInitialRuntimeStatus();
  }, [ensureInitialRuntimeStatus]);

  const isSameRuntimeStatus = React.useCallback(
    (a?: WorkflowNodeRuntimeStatus, b?: WorkflowNodeRuntimeStatus) => {
      if (a === b) return true;
      if (!a || !b) return false;
      return (
        a.slug === b.slug &&
        a.phase === b.phase &&
        a.message === b.message &&
        a.progress === b.progress &&
        a.startedAt === b.startedAt &&
        a.finishedAt === b.finishedAt
      );
    },
    []
  );

  const handleNodesInternalChange = useCallback(
    (changes: Parameters<typeof onNodesChange>[0]) => {
      onNodesChange(changes);
      const affectsCompile = changes.some(
        (change) => change.type !== "position" && change.type !== "dimensions" && change.type !== "select"
      );
      if (affectsCompile) {
        markDirty("all");
        return;
      }
      const affectsLayout = changes.some(
        (change) => change.type === "position" || change.type === "dimensions"
      );
      if (affectsLayout) {
        markDirty("layout");
      }
    },
    [markDirty, onNodesChange]
  );

  const handleEdgesInternalChange = useCallback(
    (changes: Parameters<typeof onEdgesChange>[0]) => {
      onEdgesChange(changes);
      if (changes.some((change) => change.type !== "select")) {
        markDirty("all");
      }
    },
    [markDirty, onEdgesChange]
  );

  const onConnect = useCallback(
    (params: Edge | Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            markerEnd: { type: MarkerType.ArrowClosed },
          } as Edge,
          eds
        )
      );
      markDirty("all");
    },
    [setEdges, markDirty]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  );

  React.useEffect(() => {
    if (selectedNodeId && !nodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(null);
    }
  }, [nodes, selectedNodeId]);

  React.useEffect(() => {
    if (submitting) {
      setTerminalOpen(true);
    }
  }, [submitting]);

  React.useEffect(() => {
    if (submitResult) {
      setTerminalOpen(true);
    }
  }, [submitResult]);

  React.useEffect(() => {
    if (submitError) {
      setTerminalOpen(true);
    }
  }, [submitError]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const bounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!bounds) return;

      const position = project({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      const type = event.dataTransfer.getData(DND_MIME) as NodeTypeId;
      if (!type) return;

      const labelFromDrag = event.dataTransfer.getData("text/plain");
      const labelMap: Record<NodeTypeId, string> = {
        [NODE_TYPES.nodeInput]: "Input Node",
        [NODE_TYPES.nodeDefault]: "Default Node",
        [NODE_TYPES.nodeOutput]: "Output Node",
      };

      const metaPayload = event.dataTransfer.getData(NODE_META_MIME);
      let meta: DragMetaPayload | undefined;
      try {
        meta = metaPayload ? (JSON.parse(metaPayload) as DragMetaPayload) : undefined;
      } catch {
        meta = undefined;
      }

      const tone = typeof meta?.tone === "string" ? meta.tone : undefined;
      const templateName = typeof meta?.templateName === "string" ? meta.templateName : undefined;
      const parameterKeys = Array.isArray(meta?.parameterKeys)
        ? meta.parameterKeys.filter((key: unknown): key is string => typeof key === "string")
        : undefined;
      const parameterDefaults =
        meta?.parameterDefaults && typeof meta.parameterDefaults === "object"
          ? (meta.parameterDefaults as Record<string, unknown>)
          : undefined;

      const newNodeId = getId();
      const newNode: Node<FlowNodeData> = {
        id: newNodeId,
        type,
        position,
        data: {
          label: labelFromDrag || labelMap[type] || "Node",
          ...(tone ? { tone } : {}),
          ...(templateName ? { templateName } : {}),
          ...(parameterKeys ? { parameterKeys } : {}),
          ...(parameterDefaults ? { parameterDefaults } : {}),
          ...(type === NODE_TYPES.nodeDefault
            ? { runtimeStatus: createPendingStatus(newNodeId) }
            : {}),
        },
      };

      setNodes((nds) => nds.concat(newNode));
      markDirty("all");
    },
    [markDirty, project, setNodes]
  );

  const handleSelectionChange = useCallback(
    ({ nodes: selected }: { nodes: Node<FlowNodeData>[]; edges: Edge[] }) => {
      if (selected.length) {
        setSelectedNodeId(selected[0].id);
      } else {
        setSelectedNodeId(null);
      }
    },
    []
  );

  const handleNodeDataChange = useCallback(
    (nodeId: string, updater: (data: FlowNodeData) => FlowNodeData) => {
      setNodes((prev) =>
        prev.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: updater(node.data),
              }
            : node
        )
      );
      markDirty("all");
    },
    [markDirty, setNodes]
  );

  const handleCompileClick = useCallback(async () => {
    setCompiling(true);
    setCompileError(null);
    try {
      const result = await compileWorkflow({
        sessionId,
        nodes,
        edges,
      });
      setCompiledState({
        manifest: result.manifest,
        manifestFilename: result.manifestFilename,
        nodeSlugMap: result.nodeSlugMap,
        bucket: result.bucket,
        compiledAt: new Date().toISOString(),
      });
      setIsCompileDirty(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error compilando workflow";
      setCompileError(message);
    } finally {
      setCompiling(false);
    }
  }, [edges, nodes, sessionId]);

  const applyWorkflowRecord = useCallback(
    (record: WorkflowRecord) => {
      cancelStatusPolling();
      const safeNodes =
        record.nodes?.map((node) => ({
          ...node,
          data: {
            ...node.data,
          },
        })) ?? [];
      const safeEdges = record.edges?.map((edge) => ({ ...edge })) ?? [];
      setNodes(safeNodes);
      setEdges(safeEdges);
      setSessionId(record.sessionId || generateSessionId());
      setSelectedNodeId(null);
      setSubmitError(null);
      setActiveWorkflow(record);
      const compiledInfo =
        record.compiledManifest && record.manifestFilename
          ? {
              manifest: record.compiledManifest,
              manifestFilename: record.manifestFilename,
              nodeSlugMap: record.nodeSlugMap ?? {},
              bucket: record.lastBucket ?? undefined,
              compiledAt: record.compiledAt ?? record.updatedAt,
            }
          : null;
      setCompiledState(compiledInfo);
      setIsCompileDirty(!compiledInfo);
      setHasUnsavedChanges(false);
      setSaveError(null);
      if (record.lastWorkflowName) {
        setSubmitResult({
          workflowName: record.lastWorkflowName,
          namespace: record.lastNamespace ?? "",
          nodeSlugMap: record.nodeSlugMap ?? {},
          bucket: record.lastBucket ?? "",
          key: record.lastKey ?? "",
          manifestFilename: record.lastManifestFilename ?? record.manifestFilename ?? "",
          cliOutput: record.lastCliOutput ?? undefined,
        });
      } else {
        setSubmitResult(null);
      }
      ensureInitialRuntimeStatus();
    },
    [cancelStatusPolling, ensureInitialRuntimeStatus, setEdges, setNodes]
  );

  const openSaveModal = useCallback(() => {
    setSaveForm({
      name: activeWorkflow?.name ?? "",
      description: activeWorkflow?.description ?? "",
    });
    setSaveError(null);
    setShowSaveModal(true);
  }, [activeWorkflow]);

  const closeSaveModal = useCallback(() => {
    if (!saving) {
      setShowSaveModal(false);
    }
  }, [saving]);

  const handleSaveWorkflow = useCallback(async () => {
    const trimmedName = saveForm.name.trim();
    if (!trimmedName) {
      setSaveError("El nombre es obligatorio.");
      return;
    }
    setSaveError(null);
    const normalizedName = trimmedName.toLowerCase();
    let summaries = workflowSummaries;
    if (!summaries.length) {
      try {
        summaries = await listWorkflows();
        setWorkflowSummaries(summaries);
      } catch (error) {
        const message = error instanceof Error ? error.message : "No se pudo verificar nombres duplicados.";
        setSaveError(message);
        return;
      }
    }
    const conflict = summaries.find(
      (summary) => summary.name.trim().toLowerCase() === normalizedName && summary.workflowId !== activeWorkflow?.workflowId
    );
    if (conflict && pendingOverwriteName !== normalizedName) {
      setPendingOverwriteName(normalizedName);
      setSaveNotice(`"${conflict.name}" ya existe. Guarda de nuevo si quieres sobrescribirlo.`);
      return;
    }
    setSaveNotice(null);
    const compiledSnapshot = !isCompileDirty && compiledState ? compiledState : null;
    setSaving(true);
    try {
      const payload = await saveWorkflow({
        workflowId: activeWorkflow?.workflowId,
        name: trimmedName,
        description: saveForm.description.trim() || undefined,
        sessionId,
        nodes: sanitizeNodesForSave(nodes),
        edges: sanitizeEdgesForSave(edges),
        compiledManifest: compiledSnapshot?.manifest,
        manifestFilename: compiledSnapshot?.manifestFilename,
        nodeSlugMap: compiledSnapshot?.nodeSlugMap,
      });
      setActiveWorkflow(payload);
      setHasUnsavedChanges(false);
      setPendingOverwriteName(null);
      if (showLoadModal) {
        // Refresh the list so the updated metadata appears immediately.
        void (async () => {
          try {
            const list = await listWorkflows();
            setWorkflowSummaries(list);
          } catch {
            // Silently ignore; the modal already surfaces errors on explicit refresh.
          }
        })();
      }
      setShowSaveModal(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error guardando el workflow.";
      setSaveError(message);
    } finally {
      setSaving(false);
    }
  }, [
    activeWorkflow,
    compiledState,
    edges,
    isCompileDirty,
    nodes,
    pendingOverwriteName,
    saveForm.description,
    saveForm.name,
    sessionId,
    showLoadModal,
    workflowSummaries,
  ]);

  const refreshWorkflowSummaries = useCallback(async () => {
    setSummariesLoading(true);
    setSummariesError(null);
    try {
      const list = await listWorkflows();
      setWorkflowSummaries(list);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error consultando workflows guardados.";
      setSummariesError(message);
    } finally {
      setSummariesLoading(false);
    }
  }, []);

  React.useEffect(() => {
    if (!showLoadModal) {
      return;
    }
    void refreshWorkflowSummaries();
  }, [refreshWorkflowSummaries, showLoadModal]);

  const openLoadModal = useCallback(() => {
    setSummariesError(null);
    setShowLoadModal(true);
  }, []);

  const closeLoadModal = useCallback(() => {
    if (!loadingWorkflowId) {
      setShowLoadModal(false);
    }
  }, [loadingWorkflowId]);

  const handleLoadWorkflow = useCallback(
    async (workflowId: string) => {
      setLoadingWorkflowId(workflowId);
      setSummariesError(null);
      try {
        const record = await getWorkflow(workflowId);
        applyWorkflowRecord(record);
        setShowLoadModal(false);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Error cargando el workflow.";
        setSummariesError(message);
      } finally {
        setLoadingWorkflowId(null);
      }
    },
    [applyWorkflowRecord]
  );

  const handleSendClick = useCallback(async () => {
    if (!compiledState || isCompileDirty) {
      setSubmitError("Compila el workflow antes de enviarlo.");
      setTerminalOpen(true);
      return;
    }
    cancelStatusPolling();
    resetNodeRuntimeStatus();
    setSubmitting(true);
    setSubmitError(null);
    setSubmitResult(null);
    try {
      const result = await submitWorkflow({
        sessionId,
        nodes,
        edges,
        workflowId: activeWorkflow?.workflowId,
      });
      setSubmitResult(result);
      setCompiledState((prev) =>
        prev
          ? {
              ...prev,
              nodeSlugMap: result.nodeSlugMap,
            }
          : prev
      );
      setActiveWorkflow((prev) =>
        prev
          ? {
              ...prev,
              lastWorkflowName: result.workflowName,
              lastNamespace: result.namespace,
              lastSubmittedAt: new Date().toISOString(),
              lastBucket: result.bucket,
              lastKey: result.key,
              lastManifestFilename: result.manifestFilename,
              lastCliOutput: result.cliOutput ?? null,
              nodeSlugMap: result.nodeSlugMap,
            }
          : prev
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error enviando workflow";
      setSubmitError(message);
    } finally {
      setSubmitting(false);
    }
  }, [
    activeWorkflow,
    cancelStatusPolling,
    compiledState,
    edges,
    isCompileDirty,
    nodes,
    resetNodeRuntimeStatus,
    sessionId,
  ]);

  const applyWorkflowStatus = React.useCallback(
    (slugMap: Record<string, string>, statusNodes: WorkflowStatusNodeMap | undefined) => {
      setNodes((prev) =>
        prev.map((node) => {
          const shouldTrack = node.type === NODE_TYPES.nodeDefault;
          if (!shouldTrack) {
            if (!node.data.runtimeStatus) {
              return node;
            }
            const nextData = { ...node.data };
            delete (nextData as Record<string, unknown>).runtimeStatus;
            return { ...node, data: nextData };
          }

          const slug = slugMap[node.id];
          const runtime = slug && statusNodes ? statusNodes[slug] : undefined;
          const normalizedRuntime = normalizeStatusForDisplay(runtime);
          const existing = node.data.runtimeStatus;
          const nextStatus =
            normalizedRuntime ?? existing ?? createPendingStatus(slug ?? node.id, existing);

          if (isSameRuntimeStatus(existing, nextStatus)) {
            return node;
          }

          return {
            ...node,
            data: {
              ...node.data,
              runtimeStatus: nextStatus,
            },
          };
        })
      );
    },
    [isSameRuntimeStatus, setNodes]
  );

  React.useEffect(() => {
    cancelStatusPolling();

    if (!submitResult?.workflowName) {
      statusAbortRef.current = true;
      return;
    }

    const slugMap = submitResult.nodeSlugMap ?? {};
    if (Object.keys(slugMap).length === 0) {
      statusAbortRef.current = true;
      return;
    }

    statusAbortRef.current = false;

    const pollStatus = async () => {
      if (statusAbortRef.current) {
        return;
      }

      try {
        const status = await fetchWorkflowStatus({
          workflowName: submitResult.workflowName,
          namespace: submitResult.namespace,
        });

        if (statusAbortRef.current) {
          return;
        }

        if (!status) {
          statusPollTimeoutRef.current = setTimeout(pollStatus, STATUS_POLL_INTERVAL_WAITING);
          return;
        }

        applyWorkflowStatus(slugMap, status.nodes);

        if (!status.finished) {
          statusPollTimeoutRef.current = setTimeout(pollStatus, STATUS_POLL_INTERVAL_FAST);
        }
      } catch (error) {
        if (statusAbortRef.current) {
          return;
        }
        console.error("Workflow status polling failed:", error);
        statusPollTimeoutRef.current = setTimeout(pollStatus, STATUS_POLL_INTERVAL_ERROR);
      }
    };

    pollStatus();

    return () => {
      statusAbortRef.current = true;
      cancelStatusPolling();
    };
  }, [applyWorkflowStatus, cancelStatusPolling, submitResult]);

  return (
    <div className="flex h-screen w-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-4 border-b border-indigo-100/80 bg-gradient-to-r from-white via-indigo-50/50 to-white px-5 py-3 shadow-sm">
          <div className="flex flex-1 items-center gap-4">
            <div className="rounded-lg border border-white/80 bg-white/70 px-3 py-2 shadow-inner">
              <p className="text-sm font-semibold text-gray-900">
                {activeWorkflow?.name || "Workflow sin guardar"}
              </p>
              <p className="text-xs text-gray-500">
                {activeWorkflow ? `Actualizado ${formatDateLabel(activeWorkflow.updatedAt)}` : "Sin guardar"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={openLoadModal}
                className="inline-flex items-center rounded-full border border-gray-200/80 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-gray-700 shadow-sm transition hover:border-indigo-200 hover:text-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-1"
              >
                Cargar
              </button>
              <button
                type="button"
                onClick={openSaveModal}
                className="inline-flex items-center rounded-full border border-indigo-200 bg-indigo-50 px-4 py-1.5 text-xs font-semibold uppercase tracking-wide text-indigo-700 shadow-sm transition hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-1"
              >
                Guardar
              </button>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="flex flex-wrap items-center justify-end gap-2">
              {hasUnsavedChanges && (
                <span className="rounded-full bg-sky-50 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-sky-700">
                  Cambios sin guardar
                </span>
              )}
              <span
                className={`rounded-full px-2 py-1 text-[11px] font-semibold uppercase tracking-wide ${
                  isReadyToSend ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
                }`}
              >
                {isReadyToSend
                  ? `Compilado ${formatDateLabel(compiledState?.compiledAt)}`
                  : "Compilación pendiente"}
              </span>
              {compiling && (
                <span className="text-xs font-medium uppercase tracking-wide text-indigo-500">
                  Compilando…
                </span>
              )}
              {submitting && (
                <span className="text-xs font-medium uppercase tracking-wide text-indigo-600">
                  Enviando workflow…
                </span>
              )}
              <button
                type="button"
                onClick={handleCompileClick}
                disabled={compiling}
                className="rounded border border-gray-200 bg-white px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-700 shadow-sm transition hover:border-indigo-200 hover:text-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {compiling ? "Compilando…" : "Compilar Workflow"}
              </button>
              <div className="relative">
                <button
                  type="button"
                  onClick={handleSendClick}
                  disabled={!isReadyToSend || submitting}
                  className="rounded bg-indigo-600 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white shadow-sm transition hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {submitting ? "Enviando…" : "Enviar Workflow"}
                </button>
                {showUnsyncedHint && (
                  <span
                    className="absolute -right-1.5 -top-1.5 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-amber-500 text-[9px] font-bold text-white ring-2 ring-white"
                    title="Hay cambios sin compilar desde la última compilación."
                  >
                    !
                  </span>
                )}
              </div>
            </div>
            {showUnsyncedHint && (
              <p className="flex items-center gap-1 text-[11px] font-medium text-amber-600">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-500" aria-hidden="true" />
                Cambios sin compilar; recompila antes de enviar.
              </p>
            )}
          </div>
        </div>
        {compileError && (
          <div className="border-l-4 border-rose-400 bg-rose-50 px-5 py-2 text-sm text-rose-700">
            {compileError}
          </div>
        )}
        <div className="relative flex min-h-0 flex-1">
          <div ref={reactFlowWrapper} className="flex-1 min-h-0">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={handleNodesInternalChange}
              onEdgesChange={handleEdgesInternalChange}
              onConnect={onConnect}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onSelectionChange={handleSelectionChange}
              defaultEdgeOptions={useMemo(
                () => ({
                  markerEnd: { type: MarkerType.ArrowClosed },
                }),
                []
              )}
              nodeTypes={useMemo(
                () => ({
                  [NODE_TYPES.nodeInput]: InputNode,
                  [NODE_TYPES.nodeDefault]: DefaultNode,
                  [NODE_TYPES.nodeOutput]: OutputNode,
                }),
                []
              )}
              fitView
              className="bg-white rf-instance"
            />
          </div>
          <NodeInspector
            isOpen={Boolean(selectedNode)}
            node={selectedNode}
            sessionId={sessionId}
            nodes={nodes}
            edges={edges}
            onChange={handleNodeDataChange}
          />
        </div>
        <WorkflowTerminal
          isOpen={terminalOpen}
          onToggle={() => setTerminalOpen((prev) => !prev)}
          workflowName={submitResult?.workflowName}
          namespace={submitResult?.namespace}
          nodeSlugMap={submitResult?.nodeSlugMap}
          nodes={nodes}
          submitting={submitting}
          submitResult={submitResult}
          submitError={submitError}
        />
      </div>
      {showSaveModal && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl">
            <div className="mb-4 flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Guardar workflow</h3>
                <p className="text-sm text-gray-500">
                  Especifica un nombre descriptivo para identificar el flujo.
                </p>
              </div>
              <button
                type="button"
                className="text-gray-500 transition hover:text-gray-700"
                onClick={closeSaveModal}
                disabled={saving}
              >
                ✕
              </button>
            </div>
            <label className="block text-sm font-medium text-gray-700">
              Nombre
              <input
                type="text"
                value={saveForm.name}
                onChange={(event) =>
                  setSaveForm((prev) => ({ ...prev, name: event.target.value }))
                }
                className="mt-1 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Workflow de ejemplo"
              />
            </label>
            <label className="mt-4 block text-sm font-medium text-gray-700">
              Descripción
              <textarea
                value={saveForm.description}
                onChange={(event) =>
                  setSaveForm((prev) => ({ ...prev, description: event.target.value }))
                }
                className="mt-1 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                rows={3}
                placeholder="Notas cortas sobre este grafo"
              />
            </label>
            {saveNotice && <p className="mt-2 text-sm text-amber-600">{saveNotice}</p>}
            {saveError && <p className="mt-2 text-sm text-rose-600">{saveError}</p>}
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={closeSaveModal}
                disabled={saving}
                className="rounded border border-gray-300 px-4 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-1"
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={handleSaveWorkflow}
                disabled={saving}
                className="rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-1 disabled:opacity-70"
              >
                {saving ? "Guardando…" : "Guardar"}
              </button>
            </div>
          </div>
        </div>
      )}
      {showLoadModal && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-3xl rounded-lg bg-white p-5 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Workflows guardados</h3>
                <p className="text-sm text-gray-500">
                  Selecciona un workflow para cargarlo en el lienzo actual.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => refreshWorkflowSummaries()}
                  className="rounded border border-gray-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-1"
                >
                  Actualizar
                </button>
                <button
                  type="button"
                  onClick={closeLoadModal}
                  className="text-gray-500 transition hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
            {summariesError && <p className="mb-3 text-sm text-rose-600">{summariesError}</p>}
            <div className="max-h-[420px] overflow-y-auto">
              {summariesLoading && (
                <p className="py-6 text-center text-sm text-gray-500">Cargando…</p>
              )}
              {!summariesLoading && workflowSummaries.length === 0 && (
                <p className="py-6 text-center text-sm text-gray-500">
                  Aún no hay workflows guardados.
                </p>
              )}
              <div className="grid gap-3">
                {workflowSummaries.map((summary) => (
                  <button
                    key={summary.workflowId}
                    type="button"
                    onClick={() => handleLoadWorkflow(summary.workflowId)}
                    disabled={loadingWorkflowId === summary.workflowId}
                    className="flex w-full items-center justify-between rounded border border-gray-200 px-4 py-3 text-left transition hover:border-indigo-200 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    <div>
                      <p className="text-sm font-semibold text-gray-900">{summary.name}</p>
                      {summary.description && (
                        <p className="text-xs text-gray-500">{summary.description}</p>
                      )}
                      <p className="text-xs text-gray-500">
                        Actualizado {formatDateLabel(summary.updatedAt)}
                      </p>
                      {summary.lastWorkflowName && (
                        <p className="text-xs text-gray-500">
                          Última ejecución: {summary.lastWorkflowName}
                        </p>
                      )}
                    </div>
                    <span className="text-xs font-semibold uppercase tracking-wide text-indigo-600">
                      {loadingWorkflowId === summary.workflowId ? "Cargando…" : "Cargar"}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function FlowEditor() {
  return (
    <ReactFlowProvider>
      <EditorInner />
    </ReactFlowProvider>
  );
}
