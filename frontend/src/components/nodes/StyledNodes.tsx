"use client";

import React from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import NodeCard, { type NodeTone } from "@/components/NodeCard";
import type { FlowNodeData } from "@/types/flow";

type Variant = "input" | "default" | "output";

function BaseNode({ data, selected, variant }: NodeProps<FlowNodeData> & { variant: Variant }) {
  const label = data.label || "Node";

  return (
    <div className="relative">
      {/* Handles depending on variant */}
      {variant !== "input" && (
        <Handle type="target" position={Position.Left} style={{ width: 12, height: 12 }} />
      )}
      {variant !== "output" && (
        <Handle type="source" position={Position.Right} style={{ width: 12, height: 12 }} />
      )}
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
