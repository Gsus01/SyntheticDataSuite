"use client";

import React from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import NodeCard from "@/components/NodeCard";

type Variant = "input" | "default" | "output";
type NodeData = { label?: string };

function BaseNode({ data, selected, variant }: NodeProps<NodeData> & { variant: Variant }) {
  const label = data?.label ?? "Node";

  return (
    <div className="relative">
      {/* Handles depending on variant */}
      {variant !== "input" && (
        <Handle type="target" position={Position.Left} style={{ width: 12, height: 12 }} />
      )}
      {variant !== "output" && (
        <Handle type="source" position={Position.Right} style={{ width: 12, height: 12 }} />
      )}
      <NodeCard label={label} variant={variant} selected={selected} />
    </div>
  );
}

export const InputNode = (props: NodeProps<NodeData>) => (
  <BaseNode {...props} variant="input" />
);
export const DefaultNode = (props: NodeProps<NodeData>) => (
  <BaseNode {...props} variant="default" />
);
export const OutputNode = (props: NodeProps<NodeData>) => (
  <BaseNode {...props} variant="output" />
);
