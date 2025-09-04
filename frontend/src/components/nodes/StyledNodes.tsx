"use client";

import React from "react";
import { Handle, Position, type NodeProps } from "reactflow";
import NodeCard from "@/components/NodeCard";

type Variant = "input" | "default" | "output";
type NodeData = { label?: string };

function BaseNode({ data, variant }: NodeProps<NodeData> & { variant: Variant }) {
  const label = data?.label ?? "Node";

  return (
    <div className="relative">
      {/* Handles depending on variant */}
      {variant !== "input" && (
        <Handle type="target" position={Position.Left} className="!w-3 !h-3" />
      )}
      {variant !== "output" && (
        <Handle type="source" position={Position.Right} className="!w-3 !h-3" />
      )}
      <NodeCard label={label} variant={variant} />
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
