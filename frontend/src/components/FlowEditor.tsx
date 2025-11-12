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
import { submitWorkflow, fetchWorkflowStatus, type SubmitWorkflowResult } from "@/lib/workflow";

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

const FINAL_PHASES = new Set([
  "succeeded",
  "failed",
  "error",
  "terminated",
  "skipped",
  "omitted",
]);

function isFinalPhase(phase?: string | null): boolean {
  if (!phase) {
    return false;
  }
  return FINAL_PHASES.has(phase.toLowerCase());
}

function computeDefaultExecutionOrder(nodes: Node<FlowNodeData>[], edges: Edge[]): string[] {
  const defaultNodes = nodes.filter((node) => node.type === NODE_TYPES.nodeDefault);
  const defaultIds = defaultNodes.map((node) => node.id);
  const defaultIdSet = new Set(defaultIds);

  const indegree = new Map<string, number>();
  defaultIds.forEach((id) => indegree.set(id, 0));
  const adjacency = new Map<string, Set<string>>();

  edges.forEach((edge) => {
    const { source, target } = edge;
    if (!defaultIdSet.has(source) || !defaultIdSet.has(target)) {
      return;
    }
    indegree.set(target, (indegree.get(target) ?? 0) + 1);
    const neighbors = adjacency.get(source);
    if (neighbors) {
      neighbors.add(target);
    } else {
      adjacency.set(source, new Set([target]));
    }
  });

  const queue: string[] = [];
  defaultIds.forEach((id) => {
    if ((indegree.get(id) ?? 0) === 0) {
      queue.push(id);
    }
  });

  const order: string[] = [];
  const visited = new Set<string>();
  let index = 0;
  while (index < queue.length) {
    const current = queue[index++];
    if (visited.has(current)) {
      continue;
    }
    visited.add(current);
    order.push(current);

    const neighbors = adjacency.get(current);
    if (!neighbors) {
      continue;
    }
    for (const neighbor of neighbors) {
      const nextDegree = (indegree.get(neighbor) ?? 0) - 1;
      indegree.set(neighbor, nextDegree);
      if (nextDegree === 0) {
        queue.push(neighbor);
      }
    }
  }

  defaultIds.forEach((id) => {
    if (!visited.has(id)) {
      order.push(id);
    }
  });

  return order;
}

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

function EditorInner() {
  const reactFlowWrapper = useRef<HTMLDivElement | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  const { project } = useReactFlow();
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(null);
  const sessionId = React.useMemo(() => generateSessionId(), []);
  const [submitting, setSubmitting] = React.useState(false);
  const [submitError, setSubmitError] = React.useState<string | null>(null);
  const [submitResult, setSubmitResult] = React.useState<SubmitWorkflowResult | null>(null);
  const [terminalOpen, setTerminalOpen] = React.useState(false);
  const statusPollTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const statusAbortRef = React.useRef(false);

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

  const onConnect = useCallback(
    (params: Edge | Connection) =>
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            markerEnd: { type: MarkerType.ArrowClosed },
          } as Edge,
          eds
        )
      ),
    [setEdges]
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
    },
    [project, setNodes]
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
    },
    [setNodes]
  );

  const handleSubmitClick = useCallback(async () => {
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
      });
      setSubmitResult(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error enviando workflow";
      setSubmitError(message);
    } finally {
      setSubmitting(false);
    }
  }, [cancelStatusPolling, edges, nodes, resetNodeRuntimeStatus, sessionId]);

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
          statusPollTimeoutRef.current = setTimeout(pollStatus, 1500);
          return;
        }

        setNodes((prev) => {
          const defaultInfoMap = new Map<
            string,
            {
              status: WorkflowNodeRuntimeStatus;
              slug?: string;
            }
          >();
          let hasActualRunning = false;

          let nextNodes = prev.map((node) => {
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
            const runtime = slug ? status.nodes[slug] : undefined;
            const existing = node.data.runtimeStatus;

            const nextStatus =
              runtime ?? existing ?? createPendingStatus(slug ?? node.id, existing);

            defaultInfoMap.set(node.id, { status: nextStatus, slug });
            if (nextStatus.phase?.toLowerCase() === "running") {
              hasActualRunning = true;
            }

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
          });

          if (!hasActualRunning && defaultInfoMap.size > 0) {
            const executionOrder = computeDefaultExecutionOrder(nextNodes, edges);
            let fallbackNodeId: string | null = null;

            for (const nodeId of executionOrder) {
              const info = defaultInfoMap.get(nodeId);
              if (!info) {
                continue;
              }
              const phaseLower = info.status?.phase?.toLowerCase();
              if (!isFinalPhase(phaseLower)) {
                fallbackNodeId = nodeId;
                break;
              }
            }

            if (fallbackNodeId) {
              const info = defaultInfoMap.get(fallbackNodeId);
              const fallbackSlug = info?.slug ?? slugMap[fallbackNodeId] ?? fallbackNodeId;
              nextNodes = nextNodes.map((node) => {
                if (node.id !== fallbackNodeId) {
                  return node;
                }
                const existing = node.data.runtimeStatus;
                const baseStatus = existing ?? createPendingStatus(fallbackSlug, existing);
                const runningStatus: WorkflowNodeRuntimeStatus = {
                  ...baseStatus,
                  slug: baseStatus.slug ?? fallbackSlug,
                  phase: "Running",
                };
                return {
                  ...node,
                  data: {
                    ...node.data,
                    runtimeStatus: runningStatus,
                  },
                };
              });
            }
          }

          return nextNodes;
        });

        if (!status.finished) {
          statusPollTimeoutRef.current = setTimeout(pollStatus, 2500);
        }
      } catch (error) {
        if (statusAbortRef.current) {
          return;
        }
        console.error("Workflow status polling failed:", error);
        statusPollTimeoutRef.current = setTimeout(pollStatus, 5000);
      }
    };

    pollStatus();

    return () => {
      cancelStatusPolling();
    };
  }, [cancelStatusPolling, edges, isSameRuntimeStatus, setNodes, submitResult]);

  return (
    <div className="flex h-screen w-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-end gap-3 border-b border-gray-200 bg-white px-4 py-3">
          {submitting && (
            <span className="text-xs font-medium uppercase tracking-wide text-indigo-600">
              Enviando workflow…
            </span>
          )}
          <button
            type="button"
            onClick={handleSubmitClick}
            disabled={submitting}
            className="rounded bg-indigo-600 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white shadow-sm transition hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 cursor-pointer disabled:cursor-not-allowed disabled:opacity-70"
          >
            {submitting ? "Enviando…" : "Enviar Workflow"}
          </button>
        </div>
        <div className="relative flex min-h-0 flex-1">
          <div ref={reactFlowWrapper} className="flex-1 min-h-0">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
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
