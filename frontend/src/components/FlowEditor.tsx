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
import { DND_MIME, NODE_TYPES, NODE_META_MIME, type NodeTypeId } from "@/lib/flow-const";
import type { FlowNodeData } from "@/types/flow";
import { exportWorkflowYAML } from "@/lib/workflow";

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
  const [downloading, setDownloading] = React.useState(false);
  const [downloadError, setDownloadError] = React.useState<string | null>(null);

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

      const newNode: Node<FlowNodeData> = {
        id: getId(),
        type,
        position,
        data: {
          label: labelFromDrag || labelMap[type] || "Node",
          ...(tone ? { tone } : {}),
          ...(templateName ? { templateName } : {}),
          ...(parameterKeys ? { parameterKeys } : {}),
          ...(parameterDefaults ? { parameterDefaults } : {}),
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

  const handleExportClick = useCallback(async () => {
    setDownloading(true);
    setDownloadError(null);
    try {
      await exportWorkflowYAML({
        sessionId,
        nodes,
        edges,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error generando YAML";
      setDownloadError(message);
    } finally {
      setDownloading(false);
    }
  }, [edges, nodes, sessionId]);

  return (
    <div className="flex h-screen w-full">
      <Sidebar />
      <div className="relative flex min-w-0 flex-1">
        <div ref={reactFlowWrapper} className="flex-1">
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
          onChange={handleNodeDataChange}
        />
        <div className="absolute right-4 top-4 z-10 flex flex-col items-end gap-2">
          <button
            type="button"
            onClick={handleExportClick}
            disabled={downloading}
            className={`rounded bg-indigo-600 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white shadow-sm transition hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 ${
              downloading ? "cursor-not-allowed opacity-70" : ""
            }`}
          >
            {downloading ? "Generando…" : "Exportar YAML"}
          </button>
          {downloadError && (
            <div className="rounded border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-600 shadow-sm">
              {downloadError}
            </div>
          )}
        </div>
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
