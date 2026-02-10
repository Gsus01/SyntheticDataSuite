"use client";

import React from "react";
import { createPortal } from "react-dom";
import {
  Handle,
  Position,
  type Edge,
  type NodeProps,
  useReactFlow,
} from "@xyflow/react";
import NodeCard, { type NodeTone } from "@/components/NodeCard";
import { buildApiUrl } from "@/lib/api";
import { getOutputArtifacts } from "@/lib/workflow";
import type { FlowNode, FlowNodeData, FlowNodePorts, NodeArtifactPort } from "@/types/flow";

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

type FlowNodeRuntimeContextValue = {
  sessionId: string;
  previewRefreshVersion: number;
};

type OutputNodeProps = NodeProps<FlowNode>;

const DEFAULT_FLOW_NODE_RUNTIME_CONTEXT: FlowNodeRuntimeContextValue = {
  sessionId: "",
  previewRefreshVersion: 0,
};

const FlowNodeRuntimeContext = React.createContext<FlowNodeRuntimeContextValue>(
  DEFAULT_FLOW_NODE_RUNTIME_CONTEXT
);

export function FlowNodeRuntimeProvider({
  sessionId,
  previewRefreshVersion,
  children,
}: React.PropsWithChildren<FlowNodeRuntimeContextValue>) {
  const value = React.useMemo(
    () => ({
      sessionId,
      previewRefreshVersion,
    }),
    [previewRefreshVersion, sessionId]
  );

  return (
    <FlowNodeRuntimeContext.Provider value={value}>
      {children}
    </FlowNodeRuntimeContext.Provider>
  );
}

function useFlowNodeRuntime() {
  return React.useContext(FlowNodeRuntimeContext);
}

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
  previewRefreshVersion,
}: {
  nodeId: string;
  sessionId: string;
  templateName?: string;
  previewRefreshVersion: number;
}) {
  const reactFlow = useReactFlow<FlowNode, Edge>();
  const stateRef = React.useRef<OutputPreviewState>(outputPreviewCache.get(nodeId) ?? HIDDEN_PREVIEW_STATE);
  const requestIdRef = React.useRef(0);
  const [previewState, setPreviewState] = React.useState<OutputPreviewState>(
    () => outputPreviewCache.get(nodeId) ?? HIDDEN_PREVIEW_STATE
  );

  React.useEffect(() => {
    const cached = outputPreviewCache.get(nodeId) ?? HIDDEN_PREVIEW_STATE;
    stateRef.current = cached;
    setPreviewState(cached);
  }, [nodeId]);

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
      const currentNodes = reactFlow.getNodes();
      const currentEdges = reactFlow.getEdges();
      prunePreviewCache(new Set(currentNodes.map((node) => node.id)));

      const incomingEdges = currentEdges.filter((edge) => edge.target === nodeId);
      if (!incomingEdges.length || !sessionId || !templateName) {
        setAndCacheState(HIDDEN_PREVIEW_STATE);
        return;
      }

      try {
        const artifacts = await getOutputArtifacts(
          {
            sessionId,
            nodes: currentNodes,
            edges: currentEdges,
          },
          nodeId
        );

        if (!isActive()) return;

        const imageArtifacts = artifacts.filter((artifact) =>
          isImageArtifact(artifact.contentType, artifact.key)
        );
        const imageArtifact = imageArtifacts.find((artifact) => artifact.exists) ?? imageArtifacts[0];

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
  }, [
    nodeId,
    previewRefreshVersion,
    reactFlow,
    sessionId,
    setAndCacheState,
    templateName,
  ]);

  return previewState;
}

function BaseNode({ data, selected, variant }: NodeProps<FlowNode> & { variant: Variant }) {
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

function OutputPreviewPanel({
  state,
  onOpenImage,
}: {
  state: OutputPreviewState;
  onOpenImage: () => void;
}) {
  if (state.kind === "hidden") {
    return null;
  }

  return (
    <div className="rounded-md border border-gray-300 bg-white p-2 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex h-[150px] w-full items-center justify-center overflow-hidden rounded border border-dashed border-gray-300 bg-gray-50 dark:border-gray-700 dark:bg-gray-800">
        {state.kind === "ready" ? (
          <button
            type="button"
            onClick={onOpenImage}
            className="nodrag nopan h-full w-full cursor-zoom-in focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 dark:focus:ring-offset-gray-900"
            aria-label={`Ampliar vista previa de ${state.artifactLabel}`}
            title="Ampliar vista previa"
          >
            {/* Blob URLs are generated at runtime; Next/Image optimization does not apply here. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={state.imageUrl} alt={`Vista previa de ${state.artifactLabel}`} className="h-full w-full object-contain" />
          </button>
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

function OutputNodeComponent({ data, selected, id }: OutputNodeProps) {
  const { previewRefreshVersion, sessionId } = useFlowNodeRuntime();
  const label = data.label || "Output Node";
  const resolvedPorts = resolvePorts(data, "output");
  const [isImageModalOpen, setIsImageModalOpen] = React.useState(false);
  const previewState = useOutputImagePreview({
    nodeId: id,
    sessionId,
    templateName: data.templateName,
    previewRefreshVersion,
  });
  const showPreviewPanel = previewState.kind !== "hidden";

  React.useEffect(() => {
    if (!isImageModalOpen || previewState.kind === "ready") return;
    setIsImageModalOpen(false);
  }, [isImageModalOpen, previewState]);

  React.useEffect(() => {
    if (!isImageModalOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setIsImageModalOpen(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isImageModalOpen]);

  const modalContent =
    isImageModalOpen && previewState.kind === "ready" ? (
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Vista previa ampliada de ${previewState.artifactLabel}`}
        className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 px-4 py-6 backdrop-blur-sm dark:bg-black/70"
        onClick={() => setIsImageModalOpen(false)}
      >
        <div
          className="w-full max-w-5xl rounded-lg border border-gray-200 bg-white p-3 shadow-xl dark:border-gray-700 dark:bg-gray-900"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="truncate text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
              {previewState.artifactLabel}
            </div>
            <button
              type="button"
              onClick={() => setIsImageModalOpen(false)}
              className="rounded border border-gray-300 px-2 py-1 text-xs font-semibold text-gray-600 transition hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800 dark:focus:ring-offset-gray-900"
              aria-label="Cerrar vista previa ampliada"
            >
              ✕
            </button>
          </div>
          <div className="max-h-[80vh] overflow-auto rounded border border-gray-200 bg-gray-50 p-2 dark:border-gray-700 dark:bg-gray-950/40">
            {/* Blob URLs are generated at runtime; Next/Image optimization does not apply here. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewState.imageUrl}
              alt={`Vista ampliada de ${previewState.artifactLabel}`}
              className="max-h-[76vh] w-full object-contain"
            />
          </div>
        </div>
      </div>
    ) : null;

  return (
    <>
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
          <OutputPreviewPanel
            state={previewState}
            onOpenImage={() => {
              if (previewState.kind !== "ready") return;
              setIsImageModalOpen(true);
            }}
          />
        </div>
      </div>
      {modalContent && typeof document !== "undefined" ? createPortal(modalContent, document.body) : null}
    </>
  );
}

function InputNodeComponent(props: NodeProps<FlowNode>) {
  return <BaseNode {...props} variant="input" />;
}

function DefaultNodeComponent(props: NodeProps<FlowNode>) {
  return <BaseNode {...props} variant="default" />;
}

OutputNodeComponent.displayName = "OutputNode";
InputNodeComponent.displayName = "InputNode";
DefaultNodeComponent.displayName = "DefaultNode";

export const OutputNode = React.memo(OutputNodeComponent);
export const InputNode = React.memo(InputNodeComponent);
export const DefaultNode = React.memo(DefaultNodeComponent);
