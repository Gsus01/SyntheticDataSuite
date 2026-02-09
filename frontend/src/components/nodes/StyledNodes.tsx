"use client";

import React from "react";
import {
  Handle,
  Position,
  type Edge,
  type Node,
  type NodeProps,
  useEdges,
  useNodes,
} from "reactflow";
import NodeCard, { type NodeTone } from "@/components/NodeCard";
import { buildApiUrl } from "@/lib/api";
import { getOutputArtifacts } from "@/lib/workflow";
import type { FlowNodeData, FlowNodePorts, NodeArtifactPort } from "@/types/flow";

type Variant = "input" | "default" | "output";

type HiddenPreviewState = { kind: "hidden" };
type LoadingPreviewState = { kind: "loading"; message: string };
type ErrorPreviewState = { kind: "error"; message: string };
type ReadyPreviewState = {
  kind: "ready";
  imageUrl: string;
  artifactKey: string;
  artifactLabel: string;
  contentType?: string | null;
};

type OutputPreviewState =
  | HiddenPreviewState
  | LoadingPreviewState
  | ErrorPreviewState
  | ReadyPreviewState;

type OutputNodeProps = NodeProps<FlowNodeData> & {
  sessionId: string;
};

const HIDDEN_PREVIEW_STATE: HiddenPreviewState = { kind: "hidden" };
const outputPreviewCache = new Map<string, OutputPreviewState>();
const PENDING_PREVIEW_RETRY_MS = 2000;

function buildHandleTooltip(port: { name: string; path?: string | null }, type: "target" | "source") {
  const direction = type === "source" ? "Salida" : "Entrada";
  const pathHint = port.path ? ` • ${port.path}` : "";
  return `${direction}: ${port.name}${pathHint}`;
}

function resolvePorts(data: FlowNodeData, variant: Variant): FlowNodePorts {
  const hasExplicitPorts = Boolean(data.artifactPorts);
  return {
    inputs:
      variant === "input"
        ? []
        : hasExplicitPorts
          ? data.artifactPorts?.inputs ?? []
          : [{ name: "Entrada" }],
    outputs:
      variant === "output"
        ? []
        : hasExplicitPorts
          ? data.artifactPorts?.outputs ?? []
          : [{ name: "Salida" }],
  };
}

function renderHandles(
  ports: NodeArtifactPort[],
  type: "target" | "source",
  position: Position
) {
  if (!ports.length) return null;
  const total = ports.length;
  return (
    <div
      className={`absolute ${position === Position.Left ? "-left-3" : "-right-3"} top-0 bottom-0 flex flex-col justify-center gap-3`}
    >
      {ports.map((port, index) => {
        const offset = ((index + 1) / (total + 1)) * 100;
        const tooltip = buildHandleTooltip(port, type);
        return (
          <div key={`${type}-${port.name ?? index}`} className="group relative">
            <Handle
              id={port.name}
              type={type}
              position={position}
              style={{
                width: 12,
                height: 12,
                top: `${offset}%`,
              }}
              className="!bg-white !border-2 !border-slate-300 hover:!border-indigo-400 hover:!bg-indigo-50 transition-colors shadow-sm"
              title={tooltip}
              aria-label={tooltip}
            />
            <div
              className={`pointer-events-none absolute top-1/2 -translate-y-1/2 whitespace-nowrap rounded-md bg-slate-900 text-white text-[11px] px-2 py-1 shadow-lg opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 ${
                position === Position.Left ? "right-4 origin-left" : "left-4 origin-right"
              }`}
            >
              {tooltip}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function resolveErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message || "Error obteniendo vista previa";
  }
  return "Error obteniendo vista previa";
}

function isImageArtifact(contentType?: string | null, key?: string | null): boolean {
  const normalizedType = (contentType || "").toLowerCase();
  if (normalizedType.startsWith("image/")) {
    return true;
  }

  const hasImageExtension = /\.(png|jpe?g|gif|webp|bmp|svg|tiff?)$/i.test(key || "");
  if (!hasImageExtension) {
    return false;
  }

  return normalizedType === "" || normalizedType === "application/octet-stream";
}

function serializeNodesSignature(nodes: Node<FlowNodeData>[]): string {
  const compact = nodes
    .map((node) => {
      const { runtimeStatus, artifactPorts, ...rest } = node.data;
      void runtimeStatus;
      void artifactPorts;
      return {
        id: node.id,
        type: node.type,
        data: rest,
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id));

  return JSON.stringify(compact);
}

function serializeEdgesSignature(edges: Edge[]): string {
  const compact = edges
    .map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle ?? null,
      targetHandle: edge.targetHandle ?? null,
    }))
    .sort((a, b) => a.id.localeCompare(b.id));

  return JSON.stringify(compact);
}

function writePreviewCache(nodeId: string, nextState: OutputPreviewState) {
  const previous = outputPreviewCache.get(nodeId);
  if (
    previous?.kind === "ready" &&
    (nextState.kind !== "ready" || nextState.imageUrl !== previous.imageUrl)
  ) {
    URL.revokeObjectURL(previous.imageUrl);
  }

  if (nextState.kind === "hidden") {
    outputPreviewCache.delete(nodeId);
    return;
  }

  outputPreviewCache.set(nodeId, nextState);
}

function prunePreviewCache(activeNodeIds: Set<string>) {
  for (const [cachedNodeId, cachedState] of outputPreviewCache.entries()) {
    if (activeNodeIds.has(cachedNodeId)) continue;
    if (cachedState.kind === "ready") {
      URL.revokeObjectURL(cachedState.imageUrl);
    }
    outputPreviewCache.delete(cachedNodeId);
  }
}

function useOutputImagePreview({
  nodeId,
  sessionId,
  templateName,
}: {
  nodeId: string;
  sessionId: string;
  templateName?: string;
}) {
  const nodes = useNodes<FlowNodeData>();
  const edges = useEdges();
  const nodesRef = React.useRef<Node<FlowNodeData>[]>(nodes);
  const edgesRef = React.useRef<Edge[]>(edges);
  const stateRef = React.useRef<OutputPreviewState>(outputPreviewCache.get(nodeId) ?? HIDDEN_PREVIEW_STATE);
  const requestIdRef = React.useRef(0);
  const [previewState, setPreviewState] = React.useState<OutputPreviewState>(
    () => outputPreviewCache.get(nodeId) ?? HIDDEN_PREVIEW_STATE
  );

  React.useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  React.useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  React.useEffect(() => {
    const cached = outputPreviewCache.get(nodeId) ?? HIDDEN_PREVIEW_STATE;
    stateRef.current = cached;
    setPreviewState(cached);
  }, [nodeId]);

  React.useEffect(() => {
    prunePreviewCache(new Set(nodes.map((node) => node.id)));
  }, [nodes]);

  React.useEffect(() => {
    return () => {
      writePreviewCache(nodeId, HIDDEN_PREVIEW_STATE);
    };
  }, [nodeId]);

  const setAndCacheState = React.useCallback(
    (nextState: OutputPreviewState) => {
      writePreviewCache(nodeId, nextState);
      stateRef.current = nextState;
      setPreviewState(nextState);
    },
    [nodeId]
  );

  const graphSignature = React.useMemo(
    () => `${serializeNodesSignature(nodes)}|${serializeEdgesSignature(edges)}`,
    [edges, nodes]
  );

  React.useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    let cancelled = false;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;
    const isActive = () => !cancelled && requestIdRef.current === requestId;

    const scheduleRetry = () => {
      if (retryTimeout || !isActive()) return;
      retryTimeout = setTimeout(() => {
        retryTimeout = null;
        if (!isActive()) return;
        void run();
      }, PENDING_PREVIEW_RETRY_MS);
    };

    const run = async () => {
      const incomingEdges = edgesRef.current.filter((edge) => edge.target === nodeId);
      if (!incomingEdges.length || !sessionId || !templateName) {
        setAndCacheState(HIDDEN_PREVIEW_STATE);
        return;
      }

      try {
        const artifacts = await getOutputArtifacts(
          {
            sessionId,
            nodes: nodesRef.current,
            edges: edgesRef.current,
          },
          nodeId
        );

        if (!isActive()) return;

        const imageArtifact = artifacts.find((artifact) =>
          isImageArtifact(artifact.contentType, artifact.key)
        );

        if (!imageArtifact) {
          setAndCacheState(HIDDEN_PREVIEW_STATE);
          return;
        }

        if (!imageArtifact.exists) {
          setAndCacheState({
            kind: "loading",
            message: "Esperando archivo de imagen…",
          });
          scheduleRetry();
          return;
        }

        const current = stateRef.current;
        const shouldKeepReady =
          current.kind === "ready" && current.artifactKey === imageArtifact.key;

        if (!shouldKeepReady) {
          setAndCacheState({
            kind: "loading",
            message: "Cargando imagen…",
          });
        }

        const downloadUrl = buildApiUrl("/artifacts/download");
        downloadUrl.searchParams.set("bucket", imageArtifact.bucket);
        downloadUrl.searchParams.set("key", imageArtifact.key);

        const response = await fetch(downloadUrl.toString());
        if (!isActive()) return;

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

        const headerContentType = response.headers.get("Content-Type");
        const blob = await response.blob();
        if (!isActive()) return;

        const resolvedContentType = blob.type || headerContentType || imageArtifact.contentType || null;
        if (!isImageArtifact(resolvedContentType, imageArtifact.key)) {
          setAndCacheState(HIDDEN_PREVIEW_STATE);
          return;
        }

        const objectUrl = URL.createObjectURL(blob);
        if (!isActive()) {
          URL.revokeObjectURL(objectUrl);
          return;
        }

        const fallbackLabel =
          imageArtifact.sourceArtifactName ||
          imageArtifact.inputName ||
          imageArtifact.key.split("/").pop() ||
          "image";

        setAndCacheState({
          kind: "ready",
          imageUrl: objectUrl,
          artifactKey: imageArtifact.key,
          artifactLabel: fallbackLabel,
          contentType: resolvedContentType,
        });
      } catch (error) {
        if (!isActive()) return;

        const current = stateRef.current;
        if (current.kind === "ready" || current.kind === "loading" || current.kind === "error") {
          setAndCacheState({
            kind: "error",
            message: resolveErrorMessage(error),
          });
          return;
        }

        setAndCacheState(HIDDEN_PREVIEW_STATE);
      }
    };

    void run();
    return () => {
      cancelled = true;
      if (retryTimeout) {
        clearTimeout(retryTimeout);
        retryTimeout = null;
      }
    };
  }, [graphSignature, nodeId, sessionId, setAndCacheState, templateName]);

  return previewState;
}

function BaseNode({ data, selected, variant }: NodeProps<FlowNodeData> & { variant: Variant }) {
  const label = data.label || "Node";
  const resolvedPorts = resolvePorts(data, variant);

  return (
    <div className="relative py-3">
      {renderHandles(resolvedPorts.inputs, "target", Position.Left)}
      {renderHandles(resolvedPorts.outputs, "source", Position.Right)}
      <NodeCard
        label={label}
        variant={variant}
        tone={data.tone as NodeTone | undefined}
        selected={selected}
        status={data.runtimeStatus}
      />
    </div>
  );
}

function OutputPreviewPanel({ state }: { state: OutputPreviewState }) {
  if (state.kind === "hidden") {
    return null;
  }

  return (
    <div className="rounded-md border border-gray-300 bg-white p-2 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex h-[150px] w-full items-center justify-center overflow-hidden rounded border border-dashed border-gray-300 bg-gray-50 dark:border-gray-700 dark:bg-gray-800">
        {state.kind === "ready" ? (
          // Blob URLs are generated at runtime; Next/Image optimization does not apply here.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={state.imageUrl} alt={`Vista previa de ${state.artifactLabel}`} className="h-full w-full object-contain" />
        ) : (
          <div
            className={`px-3 text-center text-[11px] leading-relaxed ${
              state.kind === "error"
                ? "text-rose-600 dark:text-rose-400"
                : "text-gray-500 dark:text-gray-400"
            }`}
          >
            {state.message}
          </div>
        )}
      </div>
      {state.kind === "ready" ? (
        <div className="mt-2 space-y-1 text-[10px] uppercase tracking-wide text-gray-500 dark:text-gray-400">
          <div className="truncate">Archivo: {state.artifactLabel}</div>
          {state.contentType ? <div className="truncate">Content-Type: {state.contentType}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

export function OutputNode({ data, selected, id, sessionId }: OutputNodeProps) {
  const label = data.label || "Output Node";
  const resolvedPorts = resolvePorts(data, "output");
  const previewState = useOutputImagePreview({
    nodeId: id,
    sessionId,
    templateName: data.templateName,
  });
  const showPreviewPanel = previewState.kind !== "hidden";

  return (
    <div className="py-3">
      <div className={showPreviewPanel ? "w-[260px] space-y-2" : undefined}>
        <div className="relative">
          {renderHandles(resolvedPorts.inputs, "target", Position.Left)}
          <NodeCard
            label={label}
            variant="output"
            tone={data.tone as NodeTone | undefined}
            selected={selected}
            status={data.runtimeStatus}
          />
        </div>
        <OutputPreviewPanel state={previewState} />
      </div>
    </div>
  );
}

export const InputNode = (props: NodeProps<FlowNodeData>) => (
  <BaseNode {...props} variant="input" />
);
export const DefaultNode = (props: NodeProps<FlowNodeData>) => (
  <BaseNode {...props} variant="default" />
);
