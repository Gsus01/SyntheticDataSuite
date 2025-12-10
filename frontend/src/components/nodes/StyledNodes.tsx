"use client";

import React from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import NodeCard, { type NodeTone } from "@/components/NodeCard";
import type { FlowNodeData, FlowNodePorts } from "@/types/flow";

type Variant = "input" | "default" | "output";

function buildHandleTooltip(port: { name: string; path?: string | null }, type: "target" | "source") {
  const direction = type === "source" ? "Salida" : "Entrada";
  const pathHint = port.path ? ` • ${port.path}` : "";
  return `${direction}: ${port.name}${pathHint}`;
}

function BaseNode({ data, selected, variant }: NodeProps<FlowNodeData> & { variant: Variant }) {
  const label = data.label || "Node";
  const hasExplicitPorts = Boolean(data.artifactPorts);
  const resolvedPorts: FlowNodePorts = {
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

  const renderHandles = (
    ports: FlowNodePorts["inputs"],
    type: "target" | "source",
    position: Position
  ) => {
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
  };

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

export const InputNode = (props: NodeProps<FlowNodeData>) => (
  <BaseNode {...props} variant="input" />
);
export const DefaultNode = (props: NodeProps<FlowNodeData>) => (
  <BaseNode {...props} variant="default" />
);
export const OutputNode = (props: NodeProps<FlowNodeData>) => (
  <BaseNode {...props} variant="output" />
);
